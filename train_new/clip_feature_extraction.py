import os
import torch
import clip
from PIL import Image
import pickle
import time
from main import DatasetfMRI
from torchvision import transforms

root_dir = '/fsx/proj-medarc/fmri/natural-scenes-dataset/data_untar/train'

outdir = '/fsx/proj-medarc/fmri/natural-scenes-dataset/clipfeat/'




# read files
onlyfiles = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
files = []
for f in onlyfiles:
    a = f.split('.')[0]
    if a not in files:
        files.append(a)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)


train_set = DatasetfMRI(root_dir=root_dir, transform_img=preprocess, returnaddress=True)
train_dl = torch.utils.data.DataLoader(train_set, batch_size=64, num_workers=8, shuffle=False)


feat_dict = {}

bs_counter = 0
batch_img = []
batch_f = []

t0 = time.time()
for i, (img, _, _, fname) in enumerate(train_dl):
    # print(img.shape)
    # print(fname)
    # exit()




    # imgfile = os.path.join(root_dir, f + '.jpg')
    # image = preprocess(Image.open(imgfile)).unsqueeze(0).to(device)  # 1 x 3 x 224 x 224
    #
    # batch_img.append(image)
    # batch_f.append(f)

    with torch.no_grad():
        image_features = model.encode_image(img.to(device))

        for j in range(len(image_features)):
            fn = fname[j]
            feat = image_features[j]
            feat_dict[fn] = feat.cpu().numpy()



    # if len(batch_img) == 32:
    #     xb = torch.cat(batch_img, 0)
    #
    #     with torch.no_grad():
    #         image_features = model.encode_image(xb)
    #
    #         for j in range(len(image_features)):
    #             feat_ = image_features[j:j + 1].cpu().numpy()
    #             fff = batch_f[j]
    #             feat_dict[fff] = feat_
    #
    #     batch_img = []
    #     batch_f = []
    #     dt = time.time() - t0
    #     print('feat extracted', dt)
    #     t0 = time.time()

    # print(image_features.shape)
    #
    print(i + 1, len(train_dl), time.time()-t0)
    t0 = time.time()
    # break

with open(os.path.join(outdir, 'clip_feat.pickle'), 'wb') as handle:
    pickle.dump(feat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

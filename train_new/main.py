import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def norm_minmax(x):
    """
    min-max normalization of numpy array
    """
    return (x - x.min()) / (x.max() - x.min())


def tensor2np_img(t):
    """
    convert image from pytorch tensor to numpy array
    """
    ti_np = t.cpu().detach().numpy().squeeze()
    ti_np = norm_minmax(ti_np)
    if len(ti_np.shape) > 2:
        ti_np = ti_np.transpose(1, 2, 0)
    return ti_np


def plot_tensor(t):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    for i in range(len(t)):
        ti_np = tensor2np_img(t[i])
        plt.subplot(1, len(t), i + 1)
        plt.imshow(ti_np)
    plt.show()

class DatasetfMRI(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/data_untar/train', transform_img=None, returnaddress=False):

        onlyfiles = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

        self.root_dir = root_dir
        self.transform_img = transform_img

        self.ra = returnaddress

        self.files = []
        for f in onlyfiles:
            a = f.split('.')[0]
            if a not in self.files:
                self.files.append(a)

        assert len(self.files) != 0, ("Data set directory is empty!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        f_nsdgeneral = os.path.join(self.root_dir, fname + '.nsdgeneral.npy')
        f_jpg = os.path.join(self.root_dir, fname + '.jpg')
        f_num_uniques = os.path.join(self.root_dir, fname + '.num_uniques.npy')
        f_wholebrain_3d = os.path.join(self.root_dir, fname + '.wholebrain_3d.npy')
        f_coco73k = os.path.join(self.root_dir, fname + '.coco73k.npy')
        f_trial = os.path.join(self.root_dir, fname + '.trial.npy')

        nsdgeneral = np.load(f_nsdgeneral)  # 3 x 15724
        nm_uniques = np.load(f_num_uniques)  # 1, #
        wholebrain_3d = np.load(f_wholebrain_3d)  # 3 x 81 x 104 x 83 # fmri of the brain for each of the trials
        coco73k = np.load(f_coco73k)  # 1, # # label of the image
        trial = np.load(f_trial)  # 1, # number between 1-3

        # image = io.imread(f_jpg)  # 256 x 256 x 3
        image = Image.open(f_jpg)  # 256 x 256 x 3

        debug = False

        if debug:
            print(nsdgeneral.shape)
            print(nm_uniques.shape)
            print(wholebrain_3d.shape)
            print(coco73k.shape)
            print(trial.shape)
            print(image.shape)

            print(nm_uniques, coco73k, trial)

            print(nsdgeneral.min(), nsdgeneral.max())

            print('===============')
            # exit()

        #
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        #
        if self.transform_img:
            image = self.transform_img(image)

        voxels = torch.from_numpy(nsdgeneral)
        coco = torch.from_numpy(coco73k
                                )
        #

        if self.ra:
            return image, voxels, coco, fname
        return image, voxels, coco


if __name__ == "__main__":

    train_transform = transforms.Compose([transforms.RandomResizedCrop(size=128),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    val_transform = transforms.Compose([transforms.Resize(size=128),
                                        transforms.
                                       RandomHorizontalFlip(),
                                        transforms.ToTensor()])

    root_dir = '/fsx/proj-medarc/fmri/natural-scenes-dataset/data_untar'
    train_set = DatasetfMRI(root_dir=os.path.join(root_dir, 'train'), transform_img=train_transform)
    test_set = DatasetfMRI(root_dir=os.path.join(root_dir, 'test'), transform_img=train_transform)

    print('train data size:', len(train_set)) # 8559
    print('test data size:', len(test_set)) # 982


    train_dl = torch.utils.data.DataLoader(train_set, batch_size=16, num_workers=8, shuffle=True)
    test_dl = torch.utils.data.DataLoader(train_set, batch_size=300, num_workers=8, shuffle=False)


    lbls = {}

    for i, (image, voxels, coco) in enumerate(train_dl): # bs x 3 x 128 x 128, bs x 3 x 15724, bs x 1
        # print(image.shape, voxels.shape, coco.shape)
        # print(coco)
        # plot_tensor([image[0], image[1], image[2]])
        for l in coco[:, 0]:
            if l not in lbls.keys():
                lbls[l.item()]=1
            else:
                lbls[l.item()]+=1
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    for k in lbls.keys():
        print(k, lbls[k])

    n = 0
    tot = 0
    for i, (image, voxels, coco) in enumerate(test_dl): # bs x 3 x 128 x 128, bs x 3 x 15724, bs x 1
        # print(image.shape, voxels.shape, coco.shape)
        # print(coco)
        # plot_tensor([image[0], image[1], image[2]])
        for l in coco[:, 0]:

            tot +=1
            if l in lbls.keys():
                n +=1

    print(tot, n, n/tot)




    exit()




        # exit()

    # print(len(train_set))
    # for i in range(len(train_set)):
    #     train_set.__getitem__(i)

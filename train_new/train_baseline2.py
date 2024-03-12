import torch
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from resnet import ResNet18
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils
from tqdm import tqdm
import wandb

wandb.login(key='72d705540a593a2db4441a16e7bb72f3b6512a09')
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="baseline1",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.1,
        # "architecture": "CNN",
        # "dataset": "CIFAR-100",
        "epochs": 100,
    }
)


class DatasetfMRI(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/data_untar/train',
                 pseudo_lbl_file='/fsx/proj-medarc/fmri/natural-scenes-dataset/clipfeat/pseudo_lbls.pickle'
                                 '',
                 transform_img=None, returnaddress=False):

        onlyfiles = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

        with open(pseudo_lbl_file, 'rb') as handle:
            self.pseudo_lbls = pickle.load(handle)

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

        lbl = self.pseudo_lbls[fname]
        #

        if self.ra:
            return image, voxels, coco, fname, lbl
        return image, voxels, coco, lbl


def train(epoch, net, optimizer, trainloader, device):
    net.train()

    am_loss1 = utils.AverageMeter()
    am_acc = utils.AverageMeter()

    prog_bar = tqdm(
        trainloader,
        ascii=True,
        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
    )

    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, data in enumerate(prog_bar):
        img, target = data[0], data[3]

        target = target.type(torch.LongTensor)

        bs = len(img)

        img, target = img.to(device), target.to(device)

        optimizer.zero_grad()

        pred = net(img)

        loss = criterion(pred, target)

        loss.backward()

        optimizer.step()

        pred_lbl = pred.argmax(1)

        acc = (pred_lbl == target).type(torch.float).mean()

        am_loss1.update(loss.mean().item())
        am_acc.update(acc.item() * 100)
        # am_loss2.update(loss_dense.mean().item())

        prog_bar.set_description(
            "Train: E{}, loss:{:2.3f}, acc:{:2.3f}".format(
                epoch, am_loss1.avg, am_acc.avg))
    prog_bar.close()

    return am_loss1.avg, am_acc.avg


def eval(epoch, net, valloader, device):
    net.eval()

    am_loss1 = utils.AverageMeter()
    am_acc = utils.AverageMeter()

    prog_bar = tqdm(
        valloader,
        ascii=True,
        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
    )

    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, data in enumerate(prog_bar):
        img, target = data[0], data[3]

        target = target.type(torch.LongTensor)

        bs = len(img)

        with torch.no_grad():
            img, target = img.to(device), target.to(device)

            pred = net(img)

            loss = criterion(pred, target)

            pred_lbl = pred.argmax(1)

            acc = (pred_lbl == target).type(torch.float).mean()

            am_loss1.update(loss.mean().item())
            am_acc.update(acc.item() * 100)
            # am_loss2.update(loss_dense.mean().item())

            prog_bar.set_description(
                "Test: E{}, loss:{:2.3f}, acc:{:2.3f}".format(
                    epoch, am_loss1.avg, am_acc.avg))
    prog_bar.close()

    return am_loss1.avg, am_acc.avg


train_transform = transforms.Compose([transforms.RandomResizedCrop(size=128),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])

val_transform = transforms.Compose([transforms.Resize(size=128),
                                    transforms.
                                   RandomHorizontalFlip(),
                                    transforms.ToTensor()])

root_dir = '/fsx/proj-medarc/fmri/natural-scenes-dataset/data_untar'

psuedo_lbl_file_train = '/fsx/proj-medarc/fmri/natural-scenes-dataset/clipfeat/pseudo_lbls_train.pickle'
psuedo_lbl_file_val = '/fsx/proj-medarc/fmri/natural-scenes-dataset/clipfeat/pseudo_lbls_val.pickle'

train_set = DatasetfMRI(root_dir=os.path.join(root_dir, 'train'), pseudo_lbl_file=psuedo_lbl_file_train,
                        transform_img=train_transform)
val_set = DatasetfMRI(root_dir=os.path.join(root_dir, 'test'), pseudo_lbl_file=psuedo_lbl_file_val,
                       transform_img=val_transform
                       )

print('train data size:', len(train_set))  # 8559
print('test data size:', len(val_set))  # 982


batch_size = 256

train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=False)

net = ResNet18(num_classes=10)
device = 'cuda:0'

net = net.to(device)

lr =  0.1
criterion = nn.CrossEntropyLoss()

print(lr)

optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_acc = 0
best_loss = 0



for e in range(200):
    train_loss, train_acc = train(e, net, optimizer, train_dl, device)
    val_loss, val_acc = eval(e, net, val_dl, device)

    if val_acc > best_acc:
        best_acc = val_acc
        state = {'net': net.state_dict()}
        print('save checkpoint: best acc: %.2f'% best_acc)
        torch.save(state, os.path.join('/fsx/proj-medarc/fmri/natural-scenes-dataset/log', 'baseline1_best.pt'))

    wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc":val_acc, "val_loss": val_loss, "lr":optimizer.param_groups[0]['lr']})
    scheduler.step()

wandb.finish()

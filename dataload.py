from torch.functional import norm
import torchio
import torch
import pickle
import torchvision.utils as vutils
import torchvision.transforms as transforms
import os
from PIL import Image

loader = transforms.Compose([
    transforms.ToTensor()])


def sv():
    #存储图像 归一化
    data = pickle.load(open("./T2PDm.pkl","rb"))
    PD = data["PD"].unsqueeze(1)
    T2 = data["T2"].unsqueeze(1)
    for i in range(PD.shape[0]):
        vutils.save_image(PD[i].unsqueeze(0),"./res/PD/%03d.png"%i,normalize=True)
        vutils.save_image(T2[i].unsqueeze(0),"./res/T2/%03d.png"%i,normalize=True)

class DataSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        
        self.imgs = []
        for i in os.listdir("res/PD/"):
            if( ".png" in i ):
                self.imgs.append(i)
        return

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return {
            "T2":loader(Image.open("res/T2/%03d.png"%idx).convert('L')).to(torch.float),
            "PD":loader(Image.open("res/PD/%03d.png"%idx).convert('L')).to(torch.float),
        }

class DatasetLoaders(torch.utils.data.Dataset):
    def __init__(self):
        self.data = pickle.load(open("./T2PDm.pkl","rb"))
        return

    def __len__(self):
        return self.data["T2"].shape[0]

    def __getitem__(self, idx):
        return {
            "T2":self.data["T2"][idx],
            "PD":self.data["PD"][idx],
        }

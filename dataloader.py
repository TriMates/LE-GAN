import torchio
import torch
import pickle
import torchvision.utils as vutils
def mk2(pos = 70):
    #70/90
    dataset = torchio.datasets.IXI(
        "./data/",
        modalities=("T2","PD"),
        download=True
    )
    data_source = {
        "T2":"",
        "PD":""
    }
    for i in range(len(dataset)):
        if(dataset[i].PD.data.shape[-1]!= 130): continue
        print(i)
        #TODO 改变维度 拼接 a.squeeze(0).transpose(2,3).transpose(1,2).contiguous().shape
        T2 = dataset[i].T2.data.squeeze(0).transpose(1,2).transpose(0,1).contiguous()[pos].unsqueeze(0)
        PD = dataset[i].PD.data.squeeze(0).transpose(1,2).transpose(0,1).contiguous()[pos].unsqueeze(0)
        if(type(data_source["PD"])== type("")):
            data_source["T2"] = T2
            data_source["PD"] = PD
        else:
            data_source["T2"] = torch.cat((data_source["T2"], T2),dim = 0)
            data_source["PD"] = torch.cat((data_source["PD"], PD),dim = 0)
        
    f= open("./T2PD_%d.pkl"%pos,"wb")
    pickle.dump(data_source, f,protocol=4)
    f.close()
    

def mk_dataset():
    dataset = torchio.datasets.IXI(
        "./data/",
        modalities=("T2","PD"),
        download=True
    )
    #重新做一个数据集

    data_source = {
        "T2":"",
        "PD":""
    }

    for i in range(len(dataset)):
        print(i)
        #TODO 改变维度 拼接 a.squeeze(0).transpose(2,3).transpose(1,2).contiguous().shape
        T2 = dataset[i].T2.data.squeeze(0).transpose(1,2).transpose(0,1).contiguous()
        PD = dataset[i].PD.data.squeeze(0).transpose(1,2).transpose(0,1).contiguous()
        if(type(data_source["PD"])== type("")):
            data_source["T2"] = T2
            data_source["PD"] = PD
        else:
            data_source["T2"] = torch.cat((data_source["T2"], T2),dim = 0)
            data_source["PD"] = torch.cat((data_source["PD"], PD),dim = 0)
        
    f= open("./T2PD.pkl","wb")
    pickle.dump(data_source, f,protocol=4)
    f.close()


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

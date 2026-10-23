import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()
df=pd.DataFrame(data.data,columns=data.feature_names)
df["label"]=data.target
df.to_csv("breast_cancer.csv",index=False)

class CsvDataset(Dataset):
    def __init__(self,filename):
        data=pd.read_csv(filename)
        self.features=torch.tensor(
            data.drop("label",axis=1).values,
            dtype=torch.float32
        )
        self.labels=torch.tensor(
            data["label"].values,
            dtype=torch.long
        )
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index],self.labels[index]

def main():
    dataset=CsvDataset("breast_cancer.csv")
    print("Dataset size:",len(dataset))
    loader=DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    features,labels=next(iter(loader))
    print("Batch features:",features.shape)
    train_size=int(0.8*len(dataset))
    val_size=len(dataset)-train_size
    train_dataset,val_dataset=random_split(
        dataset,
        [train_size,val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print("Training dataset size:",len(train_dataset))
    print("Validation dataset size:",len(val_dataset))

    for workers in [0,2]:
        test_loader=DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=workers
        )
        batches=0
        for features,labels in test_loader:
            batches+=1
        print("num_workers:",workers)
        print("Number of batches:",batches)

if __name__=="__main__":
    main()    
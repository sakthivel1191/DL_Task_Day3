import os
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.datasets import load_breast_cancer

if not os.path.exists("breast_cancer.csv"):
    data=load_breast_cancer()
    df=pd.DataFrame(data.data,columns=data.feature_names)
    df["label"]=data.target
    df.to_csv("breast_cancer.csv",index=False)

class CsvDataset:
    def __init__(self,filename):
        data=pd.read_csv(filename)
        self.features=torch.tensor(
            data.drop("label",axis=1).values,\
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
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(30,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2)
        )
    def forward(self,x):
        return self.layers(x)

def calculate_accurracy(model,loader):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for features,labels in loader:
            output=model(features)
            prediction=torch.argmax(output,dim=1)
            correct+=(prediction==labels).sum().item()
            total+=labels.size(0)
    return correct/total*100
def predict(model,features):
    model.eval()
    features=torch.tensor(
        features,dtype=torch.float32).reshape(1,30)
    with torch.no_grad():
        output=model(features)
        probabilities=torch.softmax(output,dim=1)
        label=torch.argmax(probabilities,dim=1).item()
        probability=probabilities[0,label].item()
    return label,probability
def main():
    torch.manual_seed(42)
    dataset=CsvDataset("breast_cancer.csv")
    train_size=int(0.8*len(dataset))
    val_size=len(dataset)-train_size
    train_dataset,val_dataset=random_split(
        dataset,
        [train_size,val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader=DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )
    model=MLP()
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(
        model.parameters(),lr=0.001
    )
    for epoch in range(30):
        model.train()
        total_loss=0
        for features,label in train_loader:
            optimizer.zero_grad()
            output=model(features)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print(f"Epoch{epoch+1}/30,"
              f"Loss:{total_loss/len(train_loader):.4f}")
    accuracy_before=calculate_accurracy(model,val_loader)
    print(f"\nAccuracy before saving:{accuracy_before:.2f}%")
    torch.save(model.state_dict(),"model.pt")
    print("Model saved successfully")
    loaded_model=MLP()
    loaded_model.load_state_dict(
        torch.load(
        "model.pt",
        map_location="cpu",
        weights_only=True
        )
    )
    loaded_model.eval()
    accuracy_after=calculate_accurracy(loaded_model,val_loader)
    print(f"Accuracy after loading:{accuracy_after:.2f}%")
    print(
        "same accuracy:",
        abs(accuracy_before-accuracy_after)<0.000001
    )
    sample_feature,actual_label=dataset[0]
    label,probability=predict(
        loaded_model,sample_feature.tolist()
    )
    print("\nActual label:",actual_label.item())
    print("Prediction label",label)
    print("prediction probability:",f"{probability:.4f}")
    checkpoint={
        "epoch":30,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict()
    }
    torch.save(checkpoint,"checkpoint.pt")
    print("\nFull checkpoint saved successfully")
    checkpoint=torch.load(
        "checkpoint.pt",
        map_location="cpu",
        weights_only=True
    )
    resumed_model=MLP()
    resumed_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    resumed_optimizer=torch.optim.Adam(
        resumed_model.parameters(),lr=0.001
    )
    resumed_optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    print("Resuming from epoch:",checkpoint["epoch"]+1)
    resumed_model.train()
    for features,labels in train_loader:
        resumed_optimizer.zero_grad()
        output=resumed_model(features)
        loss=criterion(output,labels)
        loss.backward()
        resumed_optimizer.step()
    print("Training resumed successfully")

if __name__=="__main__":
    main()
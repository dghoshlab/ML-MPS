#!/usr/bin/env python3
import time
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset,ConcatDataset,SubsetRandomSampler
from pandas import read_csv
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,mean_squared_error
from numpy import vstack
from mps_setup import readInput
import os
import sys

bond_dim, adaptive_mode, periodic_bc, batch_size, num_epochs, learn_rate, l2_reg, nfold, inp_dim, pth, file_name, input_list, det, TorchMPS_path = readInput()
sys.path.append(TorchMPS_path+'/TorchMPS/')
from torchmps import MPS


# Miscellaneous initialization
torch.manual_seed(0)
start_time = time.time()

# Initialize the MPS module
mps = MPS(
    input_dim=inp_dim,
    output_dim=1,
    bond_dim=bond_dim,
    adaptive_mode=adaptive_mode,
    periodic_bc=periodic_bc,
)

# Set our loss function and optimizer
loss_fun = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)

# Get the training and test sets
class CSVDataset(Dataset):
    def __init__(self, path):
        df  = read_csv(path,usecols=input_list, header=None)
        df_det = read_csv(path,usecols=[det], header=None)


        self.X = df.values[:, :-1]                           # Input descriptor #
        self.y = df.values[:, -1]                            # Output #
        self.det = df_det.values[:,-1]

        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.det = self.det.astype('float32')

        self.y = self.y.reshape((len(self.y), 1))
        self.det = self.det.reshape((len(self.det), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx], self.det[idx]]

    def get_splits(self, n_test=0.8):                              # spliting of dataset 
        test_size = int(round(n_test * len(self.X)))
        train_size = int(len(self.X) - test_size)
        return random_split(self, [train_size, test_size])
def prepare_data(path):
    Dataset = CSVDataset(path)
    train_dl, test_dl = Dataset.get_splits()
    return train_dl, test_dl
 
path = pth+file_name
train_dl, test_dl = prepare_data(path)

dataset = ConcatDataset([train_dl, test_dl])
kfold = KFold(n_splits=nfold, shuffle=True)

torch.save(mps.state_dict(), pth+"model_M"+str(bond_dim)+".pth")
f0 = open(pth+"Error_M"+str(bond_dim)+".out","w")

for fold, (test_ids, train_ids) in enumerate(kfold.split(dataset)):
    f0.write(f'FOLD {fold+1}\n')
    f0.write('--------------------------------\n')
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)
    # Define data loaders for training and testing data in this fold
    trainloader = DataLoader(dataset,batch_size=batch_size, sampler=train_subsampler)
    testloader = DataLoader(dataset,batch_size=batch_size, sampler=test_subsampler)
    mps.load_state_dict(torch.load(pth+"model_M"+str(bond_dim)+".pth"))
    f0.write(f"Epochs\tEpoch loss\n")
    for epoch_num in range(1, num_epochs + 1):
        running_loss = 0.0

        for i, (inputs, targets ,dets) in enumerate(trainloader): # For train in small space we alter the testloader and trainloader  

            optimizer.zero_grad()
            scores = mps(inputs)

            loss = loss_fun(scores, targets)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
        epoch_loss = running_loss/len(train_dl)
        f0.write(f"{epoch_num}\t{epoch_loss:.5f}\n")
    f0.write(f"{fold+1} th training ends\n")
    if (fold == nfold-1):
        torch.save(mps.state_dict(), pth+"converged_model_M"+str(bond_dim)+".pth")


    f2 = open(pth+"mps_train_output_M"+str(bond_dim)+"_F"+str(fold+1)+".out","w")
    for i, (inputs, targets ,dets) in enumerate(trainloader):
        scores = mps(inputs)
        scores = scores.detach().numpy()
        actual = targets.numpy()
        det = dets.numpy()
        for j in range(len(actual)):
            f2.write(str(int(det[j][0]))+"     "+str(10**(actual[j][0]*-1))+"   "+str(10**(scores[j][0]*-1))+"\n")



    f1 = open(pth+"mps_test_output_M"+str(bond_dim)+"_F"+str(fold+1)+".out","w")
    for i, (inputs, targets ,dets) in enumerate(testloader):
        scores = mps(inputs)
        scores = scores.detach().numpy()
        actual = targets.numpy()
        det = dets.numpy()
        for j in range(len(actual)):
            f1.write(str(int(det[j][0]))+"     "+str(10**(actual[j][0]*-1))+"   "+str(10**(scores[j][0]*-1))+"\n")


    f1.close()
    f2.close()
f0.close()
os.remove(pth+"model_M"+str(bond_dim)+".pth")

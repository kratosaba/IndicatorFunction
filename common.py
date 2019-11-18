from __future__ import absolute_import, division, print_function


import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset




print(torch.__version__)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.fc3_bn = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.fc4_bn = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, out_dim,bias=False)
        
        self.relu = nn.LeakyReLU()
        

    def forward(self, x):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))# leaky
        x = self.relu(self.fc3_bn(self.fc3(x)))
        x = self.relu(self.fc4_bn(self.fc4(x)))
        x = self.fc5(x)
        return x

class MLP1(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 256)
        self.fc4_bn = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, out_dim,bias=False)
        
        self.relu = nn.LeakyReLU()
        

    def forward(self, x):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))# leaky
        x = self.relu(self.fc3_bn(self.fc3(x)))
        x = self.relu(self.fc4_bn(self.fc4(x)))
        x = self.fc5(x)
        return x

class MLP2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 512)
        self.fc4_bn = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 512)
        self.fc5_bn = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, out_dim,bias=False)
        
        self.relu = nn.LeakyReLU()
        

    def forward(self, x):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))# leaky
        x = self.relu(self.fc3_bn(self.fc3(x)))
        x = self.relu(self.fc4_bn(self.fc4(x)))
        x = self.relu(self.fc5_bn(self.fc5(x)))
        x = self.fc6(x)
        return x

class MLP3(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc4_bn = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, out_dim,bias=False)
        
        self.relu = nn.LeakyReLU()
        

    def forward(self, x):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))# leaky
        x = self.relu(self.fc3_bn(self.fc3(x)))
        x = self.relu(self.fc4_bn(self.fc4(x)))
        x = self.fc5(x)
        return x
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output. 
        pred = x
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans) 

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

class LabelData(Dataset):
    def __init__(self, data_tensor):
        self.y = data_tensor
        
    def __getitem__(self, index):
        return (self.y[index])
    
    def __len__(self):
        return len(self.y)


def organize_sample_data(Samples,Ocupancy,num_samples,Samples_per_slice,num_batches):
    
    num_slices = int(num_samples / Samples_per_slice)
    
    Samples = np.reshape(Samples,(num_slices,Samples_per_slice,3))
    Ocupancy = np.reshape(Ocupancy,(num_slices,Samples_per_slice,1))
    
    chunch_size =  int(Samples_per_slice / num_batches)
    
    New_position = np.array([])
    New_Ocupancy =  np.array([])
    
    batch_size = int( num_slices * chunch_size)

    beging = 0
    while beging <  Samples_per_slice:
        Sample = Samples[:num_slices,beging:beging+chunch_size,:]
        Ocupan = Ocupancy[:num_slices,beging:beging+chunch_size,:]
        Sample = np.reshape(Sample, (Sample.shape[0] * Sample.shape[1],3))
        Ocupan = np.reshape(Ocupan, (Ocupan.shape[0] * Ocupan.shape[1],1))
        if beging == 0:
            New_position = Sample
            New_Ocupancy = Ocupan
        else:
            New_position =  np.append(New_position,Sample,0)
            New_Ocupancy = np.append(New_Ocupancy,Ocupan)
        
        beging += chunch_size

    return New_position, New_Ocupancy, batch_size


def load_samples(filename):
    data = []
    #Open and save data
    file = open(filename, "r") 
    
    for line in file: 
        data.append(line.split())
    file.close()
    # read number of samples per slice
    newdata = []
    
    first_line = list(map(int,data[0]))
    del data[0]
    num_samples = first_line[0]
    slice_perslice = first_line[1]

    for line in range(0,len(data)):
        if len(data[line]) == 4:
            newdata.append(data[line])
    
    for line in range(0,len(newdata)):
        newdata[line] =list(map(float,newdata[line]))

    newdata = np.asarray(newdata)
    
    sample_points = np.zeros((newdata.shape[0],3),np.float32)
    ocupancy = np.zeros((newdata.shape[0],1),np.float32)
    for i in range(0,newdata.shape[0]):
        sample_points[i][0] = newdata[i][0]
        sample_points[i][1] = newdata[i][1]
        sample_points[i][2] = newdata[i][2]
        ocupancy[i] =  newdata[i][3]
    
    print(newdata.shape, sample_points.shape,ocupancy.shape)
    
    return sample_points, ocupancy, int(num_samples) ,int(slice_perslice)

def write_objfile(filename,verts,faces): #TODO 
    f = open(filename,"w+")
    
    f.write('OFF\n')
    
    f.write(str(verts.shape[0]) + " " + str(faces.shape[0]) + " 0 \n")   
    
    for line in verts:
        f.write(" ".join(line.astype(str)) + "\n")
    
    for line in faces:
        f.write(" ".join(line.astype(str)) + "\n")
    f.close()

def write_grid(filename,grid,resolution,p_max,p_min): #TODO 
    
    f = open(filename,"w+")

    
    f.write(str(resolution) + "\n" )
    f.write(" ".join(p_max.astype(str)) + "\n")
    f.write(" ".join(p_min.astype(str)) + "\n")
    
    for i in range(0,grid.shape[0]):
        f.write(grid[i].astype(str) + "\n")
    
    f.close()
    
def loadequation(filename):
    # TODO take into account multiple contours in a plane
    data = []
    #Open and save data
    file = open(filename, "r") 
    for line in file: 
        data.append(line.split())
    file.close()

    # getting rid of empty lines
    data = list(filter(None, data))

    #getting rid of header
    
    del data[0] 
    
    plane_equations = []
    # safe only the vertices
    for line in range(0,len(data)):
        if len(data[line]) == 7 and len(data[line][6]) > 3:
            plane_equations.append([data[line][3],data[line][4],data[line][5],data[line][6]])
        elif len(data[line]) == 8 and len(data[line][6]) > 3:
            plane_equations.append([data[line][4],data[line][5],data[line][6],data[line][7]])
    # Transform it to floats
    for line in range(0,len(plane_equations)):  
        plane_equations[line] =list(map(float,plane_equations[line]))
    
    return np.asarray(plane_equations)

def orgpos(filename):
    """
    Returns an array [n, 3] n the number of vertices. 
    :param file_name: The name of the mesh file to load
    :return: An [n, 3] array of vertex positions
    """
    data = []
    #Open and save data
    file = open(filename, "r") 
    for line in file: 
        data.append(line.split())
    file.close()

    # getting rid of empty lines
    data = list(filter(None, data))

    #getting rid of header
    
    del data[0] 
    
    newdata = []
    # safe only the vertices
    for line in range(0,len(data)):
        if len(data[line]) == 3 and len(data[line][0]) > 3:
            newdata.append(data[line])
    
    
    #mapping from string to float
    for line in range(0,len(newdata)):
        newdata[line] =list(map(float,newdata[line]))
   
    # Transform it to an array
    
    orgpos = np.asarray(newdata)
    
    return orgpos

def write_octree(filename,octree,label0,cell_list): #TODO write the points and the order in a single line
    
    f = open(filename,"w+")
    
    f.write( str(octree.fullpointlist.shape[0]) + " " + str(cell_list.shape[0]) + " 2 \n")
    
    for i in range(octree.fullpointlist.shape[0]):
        f.write( " ".join(octree.fullpointlist[i].astype(str)) + " 0 "+  (octree.allisovalues[i].astype(str)) + " \n" )
        
    f.write("\n \n")
    
    for j in range(cell_list.shape[0]):
        f.write(" ".join(cell_list[j].astype(str)) +" \n")
    f.close()


def seed_everything(seed):
    if seed < 0:
        seed = np.random.randint(np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)

    return seed


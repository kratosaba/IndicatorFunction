import argparse

import meshcut
import numpy as np 
import common
import visualize
from skimage import measure

import copy


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("samples_filename", type=str, help="Slices to reconstruct")
    argparser.add_argument("equations_filename", type=str, help="File to read the equations")
    argparser.add_argument("mesh_filename", type=str, help="Reconstruction to save")
    argparser.add_argument("--scaling-todo", "-st", type=float, default=1 ,
                           help="What scaling would you like to do")
    argparser.add_argument("--num-epochs", "-e", type=int, default=1, help="Number of training epochs")
    argparser.add_argument("--num-batches", "-b", type=int, default=72500, help="Number of training batches")
    argparser.add_argument("--resolution", "-r", type=int, default=300, help="Resolution to evaluate on network")
    argparser.add_argument("--seed", "-s", type=int, default=-1)
                           

    args = argparser.parse_args()

    if args.seed > 0:
        seed = args.seed
        common.seed_everything(seed)
    else:
        seed = np.random.randint(0, 2**32-1)
        common.seed_everything(seed)
    print("Using seed %d" % seed)


    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # We load the training data 
    Samples, Ocupancy,  num_samples, Samples_per_slice = common.load_samples(args.samples_filename)
    Samples = Samples * args.scaling_todo
    
    print(np.amin(Samples),np.amax(Samples))

    x_test = torch.from_numpy(Samples.astype(np.float32)).to(device)
    y_test = torch.from_numpy(Ocupancy.astype(np.float32)).to(device)

    train_data = common.CustomDataset(x_test, y_test)

    #Separate into bartches batches_size equal to the number of points in each slice n_samples x n_samples
    train_loader = DataLoader(dataset=train_data, batch_size= args.num_batches, shuffle=True) 


    phi = common.MLP(3, 1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(phi.parameters(), lr = 0.01)
    epoch = args.num_epochs

    fit_start_time = time.time()

    for epoch in range(epoch):
        batch = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            x_train = x_batch

            y_train = y_batch

            y_pred = phi(x_batch)
            
        
            loss = criterion(y_pred.squeeze(), y_batch.squeeze())

            
            print('Batch {}: train loss: {}'.format(batch, loss.item()))    # Backward pass

            loss.backward()
            optimizer.step() # Optimizes only phi parameters
            batch+=1
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))  

    fit_end_time = time.time()
    print("Total time = %f" % (fit_end_time - fit_start_time))
    

    min = -5
    max = 5
    complexnum = 1j

    #Sample 3D space
    X,Y,Z  = np.mgrid[min:max:args.resolution *complexnum,min:max:args.resolution *complexnum,min:max:args.resolution *complexnum] 

    with torch.no_grad():
        xyz = torch.from_numpy(np.vstack([X.ravel(), Y.ravel(),Z.ravel()]).transpose().astype(np.float32)).to(device)
        
        eval_data = common.LabelData(xyz)
        labels = np.asarray([])

        #Separate into bartches batches_size equal to the number of points in each slice n_samples x n_samples
        eval_loader = DataLoader(dataset=eval_data, batch_size= args.num_batches, shuffle=False) 

        for eval_batch in eval_loader:
            phi.eval()
            label = torch.sigmoid(phi(eval_batch).to(device))
            label = label.detach().cpu().numpy().astype(np.float32)
            labels = np.append(labels,label)
    
    I = labels.reshape((np.cbrt(labels.shape[0]).astype(np.int32),np.cbrt(labels.shape[0]).astype(np.int32),np.cbrt(labels.shape[0]).astype(np.int32)))

    verts, faces, normals = measure.marching_cubes_lewiner(I,spacing=(X[1,0, 0]-X[0,0,0], Y[0,1, 0]-Y[0,0,0], Z[0,0, 1]-Z[0,0,0]))[:3]
    verts = verts - max

    visualize.vizualize_all_contours(verts,faces,args.scaling_todo,args.equations_filename)

    common.write_objfile(args.mesh_filename,verts,faces)
if __name__ == "__main__":
    main()
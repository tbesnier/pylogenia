import sys
import torch
import open3d as o3d
import os
import time
import numpy as np
import pandas as pd
import re

sys.path.insert(0,"../")

from utils import lddmm_utils, mesh_processing, viz

# torch type and device
use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
torchdtype = torch.float32

# PyKeOps counterpart
KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
KeOpsdtype = torchdtype.__str__().split(".")[1]  # 'float32'

source_dir = "../data/preprocessed/"
data_paths = [source_dir + file for file in os.listdir(source_dir) if "ipynb" not in file]
names = [re.search('../data/preprocessed/(.*)_preprocessed.ply', path).group(1) for path in data_paths]

def get_data(file):
    mesh = o3d.io.read_triangle_mesh(file)
    V, F, Rho = mesh_processing.getDataFromMesh(mesh)
    return(V,F,Rho)

list_data = []
for file in data_paths:
    list_data.append(get_data(file)[:2])
    
tab = []
for ind_s, ls in enumerate(list_data):
    C=[]
    padd = [0 for i in range(len(list_data) - ind_s)]
    for ind_t, lt in enumerate(list_data):
        if ind_s>ind_t:
        
            q0 = torch.from_numpy(ls[0]).clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
            VT = torch.from_numpy(lt[0]).clone().detach().to(dtype=torchdtype, device=torchdeviceId)
            FS = torch.from_numpy(ls[1]).clone().detach().to(dtype=torch.long, device=torchdeviceId)
            FT = torch.from_numpy(lt[1]).clone().detach().to(dtype=torch.long, device=torchdeviceId)
            sigma = torch.tensor([10], dtype=torchdtype, device=torchdeviceId)

            x, y, z = (
                q0[:, 0].detach().cpu().numpy(),
                q0[:, 1].detach().cpu().numpy(),
                q0[:, 2].detach().cpu().numpy(),
            )
            i, j, k = (
                FS[:, 0].detach().cpu().numpy(),
                FS[:, 1].detach().cpu().numpy(),
                FS[:, 2].detach().cpu().numpy(),
            )

            xt, yt, zt = (
                VT[:, 0].detach().cpu().numpy(),
                VT[:, 1].detach().cpu().numpy(),
                VT[:, 2].detach().cpu().numpy(),
            )
            it, jt, kt = (
                FT[:, 0].detach().cpu().numpy(),
                FT[:, 1].detach().cpu().numpy(),
                FT[:, 2].detach().cpu().numpy(),
            )

#####################################################################
# Define data attachment and LDDMM functional

            dataloss = lddmm_utils.lossVarifoldSurf(FS, VT, FT, lddmm_utils.GaussLinKernel(sigma=sigma))
            Kv = lddmm_utils.GaussKernel(sigma=sigma)
            loss = lddmm_utils.LDDMMloss(Kv, dataloss)

######################################################################
# Perform optimization

# initialize momentum vectors
            p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

            optimizer = torch.optim.LBFGS([p0], max_eval=10, max_iter=10)
            dist = int(loss(p0, q0).item()/1000)
            print("LDDM loss: ",ind_s, "-",ind_t, dist)
            C.append(dist)
            #print("Hausdorff distance: ", ind_s, "-",ind_t, hausdorff_distance(ls[0].detach().cpu().numpy(),lt[0].detach().cpu().numpy()))
    C = C + padd
    
    tab.append(C)
    
tab = np.array(tab)
df = pd.DataFrame(tab, index=names,columns=names)

df.to_csv("doc/results/distance_varifold.csv")
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

file_bulldog = "../data/preprocessed/bulldog_preprocessed.ply"
file_bergen = "../data/preprocessed/Canislupus_Bergen_preprocessed.ply"
file_labrador = "../data/preprocessed/Labrador_preprocessed.ply"
file_lund = "../data/preprocessed/Canislupus_Lund_preprocessed.ply"
file_MZH = "../data/preprocessed/Canislupus_MZH_preprocessed.ply"
file_oulu = "../data/preprocessed/Canislupus_Oulu_preprocessed.ply"
file_nhmo = "../data/preprocessed/NHMO_preprocessed.ply"
file_nmb = "../data/preprocessed/NMB_preprocessed.ply"

mesh_bulldog = o3d.io.read_triangle_mesh(file_bulldog)
V1, F1, Rho1 = mesh_processing.getDataFromMesh(mesh_bulldog)

mesh_bergen = o3d.io.read_triangle_mesh(file_bergen)
V2, F2, Rho2 = mesh_processing.getDataFromMesh(mesh_bergen)

mesh_labrador = o3d.io.read_triangle_mesh(file_labrador)
V3, F3, Rho3 = mesh_processing.getDataFromMesh(mesh_labrador)

mesh_lund = o3d.io.read_triangle_mesh(file_lund)
V4, F4, Rho4 = mesh_processing.getDataFromMesh(mesh_lund)

mesh_MZH = o3d.io.read_triangle_mesh(file_MZH)
V5, F5, Rho5 = mesh_processing.getDataFromMesh(mesh_MZH)

mesh_oulu = o3d.io.read_triangle_mesh(file_oulu)
V6, F6, Rho6 = mesh_processing.getDataFromMesh(mesh_oulu)

mesh_nhmo = o3d.io.read_triangle_mesh(file_nhmo)
V7, F7, Rho7 = mesh_processing.getDataFromMesh(mesh_nhmo)

mesh_nmb = o3d.io.read_triangle_mesh(file_nmb)
V8, F8, Rho8 = mesh_processing.getDataFromMesh(mesh_nmb)

sigma = torch.tensor([10], dtype=torchdtype, device=torchdeviceId)

def frechet_mean_lddmm(V1,F1,V2,F2):
    
    q0 = torch.from_numpy(V1).clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    VT = torch.from_numpy(V2).clone().detach().to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(F1).clone().detach().to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(F2).clone().detach().to(dtype=torch.long, device=torchdeviceId)

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
    
    dataloss = lddmm_utils.lossVarifoldSurf(FS, VT, FT, lddmm_utils.GaussLinKernel(sigma=sigma))
    Kv = lddmm_utils.GaussKernel(sigma=sigma)
    loss = lddmm_utils.LDDMMloss(Kv, dataloss)
    
    p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)
    q = q0.clone()
    optimizer = torch.optim.LBFGS([p0], max_eval=1, max_iter=1)
    
    def closure():
        optimizer.zero_grad()
        L = loss(p0, q0)
        print("loss from target", L.detach().cpu().numpy())
        #print("loss from source", loss(q,q0).detach().cpu().numpy())
        L.backward()
        return L

    losses = []
    for i in range(10):
        print("it ", i, ": ", end="")
        optimizer.step(closure)
        
    nt = 10
    listpq = lddmm_utils.Shooting(p0, q0, Kv)
    
    VTnp, FTnp = VT.detach().cpu().numpy(), FT.detach().cpu().numpy()
    q0np, FSnp = q0.detach().cpu().numpy(), FS.detach().cpu().numpy()
    
    dist_target, dist_source = [], []

    for t in range(1,10):

        qnp = listpq[t][1].detach().cpu().numpy()

        V_approx = np.array([qnp[:,0], qnp[:,1], qnp[:,2]]).T
        F_approx = np.array([FSnp[:,0], FSnp[:,1], FSnp[:,2]]).T
    
        V_test = torch.tensor(V_approx).detach().to(dtype=torchdtype, device=torchdeviceId)
        F_test = torch.tensor(F_approx).detach().to(dtype=torch.long, device=torchdeviceId)
    
        dist_target.append(lddmm_utils.lossVarifoldSurf(F_test, VT, FT, lddmm_utils.GaussLinKernel(sigma=sigma))(V_test).item())
        dist_source.append(lddmm_utils.lossVarifoldSurf(F_test, q0, FS, lddmm_utils.GaussLinKernel(sigma=sigma))(V_test).item())
    
    i = np.where(np.array(dist_source) - np.array(dist_target)>0)[0][0]    
    qnp = listpq[i][1].detach().cpu().numpy()

    V_mean = np.array([qnp[:,0], qnp[:,1], qnp[:,2]]).T
    F_mean = np.array([FSnp[:,0], FSnp[:,1], FSnp[:,2]]).T
    
    return(V_mean, F_mean)

V_mean, F_mean = frechet_mean_lddmm(V1,F1,V2,F2)
mesh_processing.export_mesh(V_mean, F_mean, "doc/results/result_mean.ply")
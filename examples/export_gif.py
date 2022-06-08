import sys
import torch
import open3d as o3d
import os
import time
import argparse
import numpy as np
import imageio

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0,"../")

from utils import lddmm_utils, mesh_processing, viz

file_ref = "../data/preprocessed/bulldog_preprocessed.ply"

mesh = o3d.io.read_triangle_mesh(file_ref)

source = "../data/preprocessed/bulldog_preprocessed.ply"
target = "../data/preprocessed/Labrador_preprocessed.ply"

# torch type and device
use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
torchdtype = torch.float32

# PyKeOps counterpart
KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
KeOpsdtype = torchdtype.__str__().split(".")[1]  # 'float32'

file_ref = source
file_target = target

mesh = o3d.io.read_triangle_mesh(file_ref)
VS, FS, RhoS = mesh_processing.getDataFromMesh(mesh)
VS, FS = torch.from_numpy(VS), torch.from_numpy(FS)

mesh = o3d.io.read_triangle_mesh(file_target)
VT, FT, RhoT = mesh_processing.getDataFromMesh(mesh)
VT, FT = torch.from_numpy(VT), torch.from_numpy(FT)

q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)
FS = FS.clone().detach().to(dtype=torch.long, device=torchdeviceId)
FT = FT.clone().detach().to(dtype=torch.long, device=torchdeviceId)
sigma = torch.tensor([15], dtype=torchdtype, device=torchdeviceId)

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

save_folder, name = "doc/results/", "data.html"
os.makedirs(save_folder, exist_ok=True)

viz.show_meshes(VS, FS, VT, FT, save_folder, name, auto_open=False)

dataloss = lddmm_utils.lossVarifoldSurf(FS, VT, FT, lddmm_utils.GaussLinKernel(sigma=sigma))
Kv = lddmm_utils.GaussKernel(sigma=sigma)
loss = lddmm_utils.LDDMMloss(Kv, dataloss)

p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

optimizer = torch.optim.LBFGS([p0], max_eval=10, max_iter=10)
print("performing optimization...")
start = time.time()

def closure():
    optimizer.zero_grad()
    L = loss(p0, q0)
    print("loss", L.detach().cpu().numpy())
    L.backward()
    return L

for i in range(10):
    print("it ", i, ": ", end="")
    optimizer.step(closure)

print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")

nt = 20
listpq = lddmm_utils.Shooting(p0, q0, Kv, nt=nt)

VTnp, FTnp = VT.detach().cpu().numpy(), FT.detach().cpu().numpy()
q0np, FSnp = q0.detach().cpu().numpy(), FS.detach().cpu().numpy()

viz.show_registration(VTnp, FTnp, FSnp, listpq, save_folder)

end = listpq[-1][1].detach().cpu().numpy()
V_approx = np.array([end[:,0], end[:,1], end[:,2]]).T
F_approx = np.array([FSnp[:,0], FSnp[:,1], FSnp[:,2]]).T

filenames = []
for i in range(len(listpq)):
    #create plot
    #plt.plot(y[:i])
    #plt.ylim(20,50)
    
    def frustum(left, right, bottom, top, znear, zfar):
        M = np.zeros((4, 4), dtype=np.float32)
        M[0, 0] = +2.0 * znear / (right - left)
        M[1, 1] = +2.0 * znear / (top - bottom)
        M[2, 2] = -(zfar + znear) / (zfar - znear)
        M[0, 2] = (right + left) / (right - left)
        M[2, 1] = (top + bottom) / (top - bottom)
        M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
        M[3, 2] = -1.0
        return M
    def perspective(fovy, aspect, znear, zfar):
        h = np.tan(0.5*np.radians(fovy)) * znear
        w = h * aspect
        return frustum(-w, w, -h, h, znear, zfar)
    def translate(x, y, z):
        return np.array([[1, 0, 0, x], [0, 1, 0, y],
                         [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)
    def xrotate(theta):
        t = np.pi * theta / 180
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0,  0, 0], [0, c, -s, 0],
                         [0, s,  c, 0], [0, 0,  0, 1]], dtype=float)
    def yrotate(theta):
        t = np.pi * theta / 180
        c, s = np.cos(t), np.sin(t)
        return  np.array([[ c, 0, s, 0], [ 0, 1, 0, 0],
                          [-s, 0, c, 0], [ 0, 0, 0, 1]], dtype=float)

    def zrotate(theta):
        t = np.pi * theta / 180
        c, s = np.cos(t), np.sin(t)
        return  np.array([[ c, -s, 0, 0], [ s, c, 0, 0],
                          [0, 0, 1, 0], [ 0, 0, 0, 1]], dtype=float)

    it = listpq[i][1].detach().cpu().numpy()
    V = np.array([it[:,0], it[:,1], it[:,2]]).T
    F = np.array([FSnp[:,0], FSnp[:,1], FSnp[:,2]]).T

    V = (V-(V.max(0)+V.min(0))/2) / max(V.max(0)-V.min(0))
    MVP = perspective(25,1,1,100) @ translate(0,0,-3.5) @ xrotate(120) @ yrotate(180) @ zrotate(-20)
    V = np.c_[V, np.ones(len(V))]  @ MVP.T
    V /= V[:,3].reshape(-1,1)
    V = V[F]
    T =  V[:,:,:2]
    Z = -V[:,:,2].mean(axis=1)
    zmin, zmax = Z.min(), Z.max()
    Z = (Z-zmin)/(zmax-zmin)
    C = plt.get_cmap("magma")(Z)
    I = np.argsort(Z)
    T, C = T[I,:], C[I,:]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1], aspect=1, frameon=False)
    collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="black")
    ax.add_collection(collection)
    
    # create file name and append it to a list
    filename = f'{i}.png'
    filenames.append(filename)
    
    # last frame of each viz stays longer
    if (i == len(listpq)-1):
            for i in range(5):
                filenames.append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()# build gif
with imageio.get_writer('registration_15.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)
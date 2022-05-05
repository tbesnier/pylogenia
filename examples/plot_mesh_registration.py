import sys
import torch
import open3d as o3d
import os
import time
import argparse
import numpy as np

sys.path.insert(0,"../")

from utils import lddmm_utils, mesh_processing, viz


parser = argparse.ArgumentParser(description='Arguments for file specification')
parser.add_argument('-s','--source', type=str, default= "../data/preprocessed/bulldog_preprocessed.ply",
                help='source mesh path')
parser.add_argument('-v', '--target', type=str, default= "../data/preprocessed/Labrador_preprocessed.ply",
                        help='target mesh path')

args = parser.parse_args()

source = args.source
target = args.target

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

for i in range(2):
    print("it ", i, ": ", end="")
    optimizer.step(closure)

print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")

nt = 10
listpq = lddmm_utils.Shooting(p0, q0, Kv, nt=nt)

VTnp, FTnp = VT.detach().cpu().numpy(), FT.detach().cpu().numpy()
q0np, FSnp = q0.detach().cpu().numpy(), FS.detach().cpu().numpy()

viz.show_registration(VTnp, FTnp, FSnp, listpq, save_folder)

end = listpq[-1][1].detach().cpu().numpy()
V_approx = np.array([end[:,0], end[:,1], end[:,2]]).T
F_approx = np.array([FSnp[:,0], FSnp[:,1], FSnp[:,2]]).T

mesh_processing.export_mesh(V_approx, F_approx, "doc/results/result.ply")
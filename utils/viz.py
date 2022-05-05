import torch
from torch.autograd import grad

import plotly.graph_objs as go

import open3d as o3d
import trimesh as tri


def show_meshes(VS, FS, VT, FT, save_folder, name, auto_open=False):
    """Create an html file containing the plot
    """
    x, y, z = (
        VS[:, 0].detach().cpu().numpy(),
        VS[:, 1].detach().cpu().numpy(),
        VS[:, 2].detach().cpu().numpy(),
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
    fig = go.Figure(
        data=[
            go.Mesh3d(x=xt, y=yt, z=zt, i=it, j=jt, k=kt, color="blue", opacity=0.50),
            go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="red", opacity=0.50),
        ]
    )
    fig.write_html(save_folder + name, auto_open=auto_open)   
    
    
def show_registration(VTnp, FTnp, FSnp, listpq, save_folder, auto_open = False):
    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            visible=True,
            x=VTnp[:, 0],
            y=VTnp[:, 1],
            z=VTnp[:, 2],
            i=FTnp[:, 0],
            j=FTnp[:, 1],
            k=FTnp[:, 2],
            opacity = 0.5
        )
    )

    # Add traces, one for each slider step
    for t in range(len(listpq)):
        qnp = listpq[t][1].detach().cpu().numpy()
        fig.add_trace(
            go.Mesh3d(
                visible=False,
                x=qnp[:, 0],
                y=qnp[:, 1],
                z=qnp[:, 2],
                i=FSnp[:, 0],
                j=FSnp[:, 1],
                k=FSnp[:, 2],
                opacity = 0.5
            )
        )

    # Make 10th trace visible
    fig.data[1].visible = True
    
    # Create and add slider
    steps = []
    for i in range(len(fig.data) - 1):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step["args"][1][0] = True
        step["args"][1][i + 1] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    
    sliders = [
        dict(active=0, currentvalue={"prefix": "time: "}, pad={"t": 20}, steps=steps)
    ]
    
    fig.update_layout(sliders=sliders)
    
    fig.write_html(save_folder + "registration_result.html", auto_open=auto_open) 
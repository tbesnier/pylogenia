import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import os
import open3d as o3d
import copy

def decimate_mesh(V,F,target):    
    """
    Decimates mesh given by V,F to have number of faces approximatelyu equal to target 
    """
    mesh=getMeshFromData([V,F])
    mesh=mesh.simplify_quadric_decimation(target)
    VS = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    FS = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array
    return VS, FS    
    
def subdivide_mesh(V,F,Rho=None,order=1):    
    """
    Performs midpoint subdivision. Order determines the number of iterations
    """
    mesh=getMeshFromData([V,F],Rho=Rho)
    mesh = mesh.subdivide_midpoint(number_of_iterations=order)
    VS = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    FS = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array
    if Rho is not None:
        RhoS = np.asarray(mesh.vertex_colors,np.float64)[:,0]
        return VS, FS, RhoS
   
    return VS, FS  

def getDataFromMesh(mesh):    
    """
    Get vertex and face connectivity of a mesh
    """
    V = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    F = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array  
    color=np.zeros((int(np.size(V)/3),0))
    if mesh.has_vertex_colors():
        color=np.asarray(255*np.asarray(mesh.vertex_colors,dtype=np.float64), dtype=np.int)
    return V, F, color
    
def getMeshFromData(mesh,Rho=None, color=None):    
    """
    Performs midpoint subdivision. Order determines the number of iterations
    """
    V=mesh[0]
    F=mesh[1] 

    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),o3d.utility.Vector3iVector(F))
    
    if Rho is not None:
        Rho=np.squeeze(Rho)
        col=np.stack((Rho,Rho,Rho))
        mesh.vertex_colors =  o3d.utility.Vector3dVector(col.T)
        
    if color is not None:
        mesh.vertex_colors =  o3d.utility.Vector3dVector(color)   
    return mesh
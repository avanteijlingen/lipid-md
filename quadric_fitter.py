#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:30:15 2020

@author: Chris
"""


import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import shgo
from scipy import linalg

import glob
import MDAnalysis

def PCA(data):
    '''
    Perform Principal Component Analysis on a point cloud.
    Subsequently transform the point cloud to the origin and so that it lies 
    in the frame of principal components. 
    '''
    #centering the data
    data -= np.mean(data, axis = 0)  
    
    cov = np.cov(data, rowvar = False)
    try:
        evals , evecs = linalg.eigh(cov)
        
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        
        a = np.dot(data, evecs) 
    
        return a, evals, evecs
    
    except ValueError:
        return 0
    

def fun(paras, x, y, z):
    
    result = 0
    for i in range(len(x)):
        result += ((paras[0]*x[i]**2) + (paras[1]*y[i]**2) + (paras[2]*x[i]*y[i])
                   +(paras[3]*x[i]) + (paras[4]*y[i]) + paras[5] - (z[i]))**2

    v = result**0.5

    return v

def quadric(paras, x, y, z):

    t_1 = paras[0]*x**2
    t_2 = paras[1]*y**2 
    t_3 = paras[2]*x*y  
    t_4 = paras[3]*x 
    t_5 = paras[4]*y 
    t_6 = paras[5]
    t_7 = z
    
    t = t_1 + t_2 + t_3 + t_4 + t_5 + t_6 - t_7
    
    return t
    

def gaussian_curvature(paras):
    
    E = 1 + (paras[3]**2)
    F = paras[3]*paras[4]
    G = 1 + (paras[4]**2)
    
    L = 2*paras[0]
    M = paras[2]
    N = 2*paras[1]
    
    nom = (L*N) - (M**2)
    den = (E*G) - (F**2)
    
    K = nom/den
    
    return K

def mean_curvature(paras):
    
    E = 1 + (paras[3]**2)
    F = paras[3]*paras[4]
    G = 1 + (paras[4]**2)
    
    L = 2*paras[0]
    M = paras[2]
    N = 2*paras[1]

    nom = (E*N) - (2*F*M) + (G*L)
    den = (E*G) - (F**2)
    
    H = nom/den
    
    return H


def fitting(a, index, file, cut_off_radius, bead):

        
    '''
    x0 is the initial guess array for the parameters of the quadric function
    
    x0[0] = P
    x0[1] = Q
    x0[2] = R
    x0[3] = S
    x0[4] = T
    x0[5] = C
    x0[6] = U
    
    '''
    b = [(-10, 10),(-10, 10),(-10, 10),(-10, 10),(-10, 10),(-10, 10)]

    x = a[0][0:,0]
    y = a[0][0:,1]
    z = a[0][0:,2]
    
    #perform a least squares fit of the quadric form to the point cloud
    res= shgo(fun, b, args = (x,y,z))
    
    # print(res_lsq)
    
    #calculate the gaussian curvature from the fit of the parameters
    valK = gaussian_curvature(res.x)
    valH = mean_curvature(res.x)
    success = res.success
    eval_val = res.fun
    
    
    return valK, valH, success, eval_val


def get_surrounding_coords(tree, coords, index, cut_off_radius):

    surrounding_indicies = tree.query_ball_point(coords[index], cut_off_radius)
    
    surrounding_coords = coords[surrounding_indicies]
    
    return surrounding_coords


def file_reader(file, bead, wrap = False):
    #pipeline = ov.io.import_file(file)
    pipeline = MDAnalysis.Universe(file)
    
    if wrap == True:
        #pipeline.modifiers.append(ov.modifiers.WrapPeriodicImagesModifier())
        print("NO WRAP FUNCTION")
    
    #pipeline.modifiers.append(ov.modifiers.SelectTypeModifier(property = 'Particle Type', types = set(bead)))    
    
    #data = pipeline.compute()
    data  = pipeline.select_atoms("name PO4")
    
    #a = np.where(data.particles.selection[:]==1)[0]

    #pos = np.array(data.particles.positions[:][a])
    pos = data.positions

    # b = list(bead)
    # c = ''.join(b)

    # fname = file.split('.pdb')[0]+'_'+c+'_coords.p'

    # pickle.dump(pos, open(fname, 'wb'))
    
    return pos

def coord_handling(file, cut_off_radius, bead):

    coords = file_reader(file, bead)
    
    tree = KDTree(coords)
    
    K_vals = []
    H_vals = []
    successes = []
    funs = []
    
    for index in range(coords.shape[0]):
        # print(file, index, coords.shape[0])
        
        #find the coordinates within a cutoff radius to form a point cloud.
        surrounding_coords = get_surrounding_coords(tree, coords, index, cut_off_radius)
        
        
        
        '''
        perform PCA on the patch in order to calculate the principal axes 
        of the point cloud. The points will then be transformed to lie in 
        the frame of the principal axes
        '''
        
        a = PCA(surrounding_coords)
        
        
        if type(a) == tuple:

            K_, H_, S_, F_ = fitting(a, index, file, cut_off_radius, bead)
            
            K_vals.append(K_)
            H_vals.append(H_)
            successes.append(S_)
            funs.append(F_)
    print(K_vals)

    d = {'K': K_vals,
         'H': H_vals,
         'Status': successes,
         'Function Value': funs}
    
    return d
    
    '''
    for calculating bending modulus:
        the pivotal plane plays a role in determining the distance between lipid pairs.
        So: when using this data, find coords of terminal carbons, headgroup carbons, and C1* beads
        - lipid vector is then the HG-terminal coords,
        - distance between lipids is the distance at the pivotal plane.
        
        this is following the method of Johner et al.  J. Phys. Chem. Lett, (2014) (see the SI)
    
    '''
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cut_off = 20
    
    pdbs = glob.glob("../../../OneDrive - University of Strathclyde/covid19/Data/Testing/*/*eq_centred.gro")
    for i, pdb in enumerate(pdbs):
        pep = pdb.split("\\")[-1].split("_eq_")[0]
        d = coord_handling(pdb, cut_off, "PO4")
        plt.scatter([i], [sum(d["K"])/len(d["K"])], label=pep)
    plt.title("Cut-off: " + str(cut_off))
    plt.legend()
    

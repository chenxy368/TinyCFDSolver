# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 03:12:39 2022

@author: HP
"""
import numpy as np
from utils.boundary import DirichletBoundary, NeumannBoundary, \
                           LinearCombinationCondition, \
                           TwoQuantityLinearCombinationCondition

boundary_classes = {"Dirichlet": DirichletBoundary,
                    "Neumann": NeumannBoundary, 
                    "LinearCombination": LinearCombinationCondition, 
                    "TwoQuantityLinearCombination": TwoQuantityLinearCombinationCondition}


class GridLoader():
    def __init__(self, root: str):
        self.root = root
        
    def create_boundary(self, boundary_information: list, mesh: np.ndarray):
        assert len(boundary_information) > 3
        
        boundary_name = boundary_information[0] + "_" + boundary_information[2]
        boundary_id = int(boundary_information[1])
        boundary_domain = np.where(mesh == int(boundary_information[1]))

        if boundary_information[3] not in boundary_classes:
            raise RuntimeError('Boundary Type Error')
            
        new_boundary = boundary_classes[boundary_information[3]](boundary_id, boundary_name, boundary_domain, boundary_information[4:])
        
        print(new_boundary)
        
        return new_boundary

class BlendSchemeGridLoader2D(GridLoader):
    def __init__(self, root: str):
        self.root = root
        
        self.interior, self.blend_interior, self.exterior, self.u_boundaries, self.v_boundaries, \
        self.psi_boundaries, self.wu_boundaries, self.wv_boundaries, self.wx_plus_boundaries, self.wy_plus_boundaries, self.wx_minus_boundaries, \
        self.wy_minus_boundaries, self.shape= self.load_grid(root)

    def load_grid(self, root: str):
        mesh = np.flipud(np.loadtxt(root + "/mesh.csv", delimiter=",", dtype = int)).transpose()
        blend_mesh = np.flipud(np.loadtxt(root + "/blend_scheme_mesh.csv", delimiter=",", dtype = int)).transpose()
        
        interior_id = -1
        blend_interior_id = 0

        u_boundaries = []
        v_boundaries = []
        psi_boundaries = []
        wu_boundaries = []
        wv_boundaries = []
        
        wx_plus_boundaries = []
        wy_plus_boundaries = []
        wx_minus_boundaries = []
        wy_minus_boundaries = []
        
        with open(root + "/blend_scheme_information.txt") as file:
            for line in file:
                str_list = line.split()
                if len(str_list) < 2:
                    raise RuntimeError('Grid Information Format Error')
                elif len(str_list) == 2 and str_list[0] == "INTERIOR":
                    blend_interior_id = int(str_list[1])
                else:
                    if str_list[0] == "wx_plus":
                        wx_plus_boundaries.append(self.create_boundary(str_list, blend_mesh))
                    elif str_list[0] == "wy_plus":
                        wy_plus_boundaries.append(self.create_boundary(str_list, blend_mesh))
                    elif str_list[0] == "wx_minus":
                        wx_minus_boundaries.append(self.create_boundary(str_list, blend_mesh))
                    elif str_list[0] == "wy_minus":
                        wy_minus_boundaries.append(self.create_boundary(str_list, blend_mesh))    
                    else:
                        raise RuntimeError('Unknown Physical Quantity')
            
        with open(root + "/grid_information.txt") as file:
            for line in file:
                str_list = line.split()
                if len(str_list) < 2:
                    raise RuntimeError('Grid Information Format Error')
                elif len(str_list) == 2 and str_list[0] == "INTERIOR":
                    interior_id = int(str_list[1])
                else:
                    if str_list[0] == "u":
                        u_boundaries.append(self.create_boundary(str_list, mesh))
                    elif str_list[0] == "v":
                        v_boundaries.append(self.create_boundary(str_list, mesh))
                    elif str_list[0] == "psi":
                        psi_boundaries.append(self.create_boundary(str_list, mesh))
                    elif str_list[0] == "w_v_psi":
                        wv_boundaries.append(self.create_boundary(str_list, mesh))   
                    elif str_list[0] == "w_u_psi":
                        wu_boundaries.append(self.create_boundary(str_list, mesh))                           
                    else:
                        raise RuntimeError('Unknown Physical Quantity')
        
        interior = np.where(mesh == interior_id) 
        blend_interior = np.where(blend_mesh == blend_interior_id)
        exterior = np.where(mesh != interior_id) 
        
        return interior, blend_interior, exterior, u_boundaries, v_boundaries, psi_boundaries, wu_boundaries, \
            wv_boundaries, wx_plus_boundaries, wy_plus_boundaries, wx_minus_boundaries, wy_minus_boundaries, mesh.shape

class StaggeredGridLoader(GridLoader):
    def __init__(self, root: str):
        self.root = root
        
        self.u_interior, self.v_interior, self.p_interior, self.u_exterior, self.v_exterior, \
        self.p_exterior, self.u_boundaries, self.v_boundaries, self.p_boundaries, \
        self.u_shape, self.v_shape, self.p_shape = self.load_grid(root)

    def load_grid(self, root: str):
        u_mesh = np.flipud(np.loadtxt(root + "/u_mesh.csv", delimiter=",", dtype = int)).transpose()
        v_mesh = np.flipud(np.loadtxt(root + "/v_mesh.csv", delimiter=",", dtype = int)).transpose()
        p_mesh = np.flipud(np.loadtxt(root + "/p_mesh.csv", delimiter=",", dtype = int)).transpose()
        
        u_interior_id = -1
        v_interior_id = -1
        p_interior_id = -1

        u_boundaries = []
        v_boundaries = []
        p_boundaries = []
        with open(root + "/grid_information.txt") as file:
            for line in file:
                str_list = line.split()
                if len(str_list) < 3:
                    raise RuntimeError('Grid Information Format Error')
                elif len(str_list) == 3 and str_list[2] == "INTERIOR":
                    if str_list[0] == "u":
                        u_interior_id = int(str_list[1])
                    elif str_list[0] == "v":
                        v_interior_id = int(str_list[1])
                    elif str_list[0] == "p":
                        p_interior_id = int(str_list[1])
                    else:
                        raise RuntimeError('Unknown Physical Quantity')
                else:
                    if str_list[0] == "u":
                        u_boundaries.append(self.create_boundary(str_list, u_mesh))      
                    elif str_list[0] == "v":
                        v_boundaries.append(self.create_boundary(str_list, v_mesh))
                    elif str_list[0] == "p":
                        p_boundaries.append(self.create_boundary(str_list, p_mesh))
                    else:
                        raise RuntimeError('Unknown Physical Quantity')
        
        interior_u = np.where(u_mesh == u_interior_id) 
        interior_v = np.where(v_mesh == v_interior_id) 
        interior_p = np.where(p_mesh == p_interior_id) 
        
        interior_else_u = np.where(u_mesh != u_interior_id) 
        interior_else_v = np.where(v_mesh != v_interior_id) 
        interior_else_p = np.where(p_mesh != p_interior_id) 
      
        return interior_u, interior_v, interior_p, interior_else_u, interior_else_v, interior_else_p, \
               u_boundaries, v_boundaries, p_boundaries, u_mesh.shape, v_mesh.shape, p_mesh.shape
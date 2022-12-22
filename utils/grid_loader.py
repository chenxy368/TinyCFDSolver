# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 03:12:39 2022

@author: HP
"""
import numpy as np
from utils.boundary import DirichletBoundary, NeumannBoundary, dirichlet_prototype, neumann_prototype, \
                           linear_combination_prototype, LinearCombinationCondition, two_quantity_linear_combination_prototype, \
                           TwoQuantityLinearCombinationCondition

class BlendSchemeGridLoader2D():
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
                        if str_list[3] == "LinearCombination":
                            offsets = []
                            coefficients = []
                            for i in range(int(str_list[4])):
                                coefficients.append(float(str_list[i*3+5]))
                                offsets.append((int(str_list[i*3+6]), int(str_list[i*3+7])))
                                
                            wx_plus_boundaries.append(LinearCombinationCondition(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                    np.where(blend_mesh == int(str_list[1])), "LinearCombination", linear_combination_prototype, \
                                                    float(str_list[len(str_list)-1]),  offsets, coefficients))
                        else:
                            raise RuntimeError('Boundary Format Error')
                    elif str_list[0] == "wy_plus":
                        if str_list[3] == "LinearCombination":
                            offsets = []
                            coefficients = []
                            for i in range(int(str_list[4])):
                                coefficients.append(float(str_list[i*3+5]))
                                offsets.append((int(str_list[i*3+6]), int(str_list[i*3+7])))
                            wy_plus_boundaries.append(LinearCombinationCondition(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                    np.where(blend_mesh == int(str_list[1])), "LinearCombination", linear_combination_prototype, \
                                                    float(str_list[len(str_list)-1]),  offsets, coefficients))
                        else:
                            raise RuntimeError('Boundary Format Error')
                    elif str_list[0] == "wx_minus":
                        if str_list[3] == "LinearCombination":
                            offsets = []
                            coefficients = []
                            for i in range(int(str_list[4])):
                                coefficients.append(float(str_list[i*3+5]))
                                offsets.append((int(str_list[i*3+6]), int(str_list[i*3+7])))
                            wx_minus_boundaries.append(LinearCombinationCondition(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                    np.where(blend_mesh == int(str_list[1])), "LinearCombination", linear_combination_prototype, \
                                                    float(str_list[len(str_list)-1]),  offsets, coefficients))
                        else:
                            raise RuntimeError('Boundary Format Error')
                    elif str_list[0] == "wy_minus":
                        if str_list[3] == "LinearCombination":
                            offsets = []
                            coefficients = []
                            for i in range(int(str_list[4])):
                                coefficients.append(float(str_list[i*3+5]))
                                offsets.append((int(str_list[i*3+6]), int(str_list[i*3+7])))
                            wy_minus_boundaries.append(LinearCombinationCondition(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                    np.where(blend_mesh == int(str_list[1])), "LinearCombination", linear_combination_prototype, \
                                                    float(str_list[len(str_list)-1]),  offsets, coefficients))
                        else:
                            raise RuntimeError('Boundary Format Error')                    
                    else:
                        print(str_list[0])
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
                        if str_list[3] == "Dirichlet" and len(str_list) == 5:
                            u_boundaries.append(DirichletBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(mesh == int(str_list[1])), "Dirichlet", dirichlet_prototype, \
                                                    float(str_list[4])))
                            print(u_boundaries[len(u_boundaries) - 1])
                        elif str_list[3] == "Neumann" and len(str_list) == 8:
                            u_boundaries.append(NeumannBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(mesh == int(str_list[1])), "Neumann", neumann_prototype, \
                                                float(str_list[4]), (int(str_list[5]), int(str_list[6])), float(str_list[7])))
                            print(u_boundaries[len(u_boundaries) - 1])
                        else:
                            raise RuntimeError('Boundary Format Error')
                    elif str_list[0] == "v":
                        if str_list[3] == "Dirichlet" and len(str_list) == 5:
                            v_boundaries.append(DirichletBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(mesh == int(str_list[1])), "Dirichlet", dirichlet_prototype, \
                                                    float(str_list[4])))
                            print(v_boundaries[len(v_boundaries) - 1])
                        elif str_list[3] == "Neumann" and len(str_list) == 8:
                            v_boundaries.append(NeumannBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(mesh == int(str_list[1])), "Neumann", neumann_prototype, \
                                                float(str_list[4]), (int(str_list[5]), int(str_list[6])), float(str_list[7])))
                            print(v_boundaries[len(v_boundaries) - 1])
                        else:
                            raise RuntimeError('Boundary Format Error')
                    elif str_list[0] == "psi":
                        if str_list[3] == "Dirichlet" and len(str_list) == 5:
                            psi_boundaries.append(DirichletBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(mesh == int(str_list[1])), "Dirichlet", dirichlet_prototype, \
                                                    float(str_list[4])))
                        elif str_list[3] == "Neumann" and len(str_list) == 8:
                            psi_boundaries.append(NeumannBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(mesh == int(str_list[1])), "Neumann", neumann_prototype, \
                                                float(str_list[4]), (int(str_list[5]), int(str_list[6])), float(str_list[7])))
                        else:
                            raise RuntimeError('Boundary Format Error')
                        print(psi_boundaries[len(psi_boundaries) - 1])
                    elif str_list[0] == "w_v_psi":
                        if str_list[3] == "TwoQuantityLinearCombination":
                            offsets1 = []
                            coefficients1 = []
                            for i in range(int(str_list[4])):
                                coefficients1.append(float(str_list[i*3+5]))
                                offsets1.append((int(str_list[i*3+6]), int(str_list[i*3+7])))
                            offsets2 = []
                            coefficients2 = []
                            for i in range(int(str_list[5 + 3 * int(str_list[4])])):
                                coefficients2.append(float(str_list[i*3 + 6 + 3 * int(str_list[4])]))
                                offsets2.append((int(str_list[i*3 + 7 + 3 * int(str_list[4])]), int(str_list[i*3 + 8 + 3 * int(str_list[4])])))
                                
                            wv_boundaries.append(TwoQuantityLinearCombinationCondition(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(mesh == int(str_list[1])), "TwoQuantityLinearCombination", \
                                                two_quantity_linear_combination_prototype, float(str_list[len(str_list)-1]),  offsets1, coefficients1, \
                                                offsets2, coefficients2))
                        else:
                            raise RuntimeError('Boundary Format Error')
                        print(wv_boundaries[len(wv_boundaries) - 1])       
                    elif str_list[0] == "w_u_psi":
                        if str_list[3] == "TwoQuantityLinearCombination":
                            offsets1 = []
                            coefficients1 = []
                            for i in range(int(str_list[4])):
                                coefficients1.append(float(str_list[i*3+5]))
                                offsets1.append((int(str_list[i*3+6]), int(str_list[i*3+7])))
                            offsets2 = []
                            coefficients2 = []
                            for i in range(int(str_list[5 + 3 * int(str_list[4])])):
                                coefficients2.append(float(str_list[i*3 + 6 + 3 * int(str_list[4])]))
                                offsets2.append((int(str_list[i*3 + 7 + 3 * int(str_list[4])]), int(str_list[i*3 + 8 + 3 * int(str_list[4])])))
                                
                            wu_boundaries.append(TwoQuantityLinearCombinationCondition(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(mesh == int(str_list[1])), "TwoQuantityLinearCombination", \
                                                two_quantity_linear_combination_prototype, float(str_list[len(str_list)-1]),  offsets1, coefficients1, \
                                                offsets2, coefficients2))
                        else:
                            raise RuntimeError('Boundary Format Error')
                        print(wu_boundaries[len(wu_boundaries) - 1])                           
                    else:
                        raise RuntimeError('Unknown Physical Quantity')
        
        interior = np.where(mesh == interior_id) 
        blend_interior = np.where(blend_mesh == blend_interior_id)
        exterior = np.where(mesh != interior_id) 
        
        return interior, blend_interior, exterior, u_boundaries, v_boundaries, psi_boundaries, wu_boundaries, \
            wv_boundaries, wx_plus_boundaries, wy_plus_boundaries, wx_minus_boundaries, wy_minus_boundaries, mesh.shape

class StaggeredGridLoader():
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
                        if str_list[3] == "Dirichlet" and len(str_list) == 5:
                            u_boundaries.append(DirichletBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(u_mesh == int(str_list[1])), "Dirichlet", dirichlet_prototype, \
                                                    float(str_list[4])))
                            print(u_boundaries[len(u_boundaries) - 1])
                        elif str_list[3] == "Neumann" and len(str_list) == 8:
                            u_boundaries.append(NeumannBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(u_mesh == int(str_list[1])), "Neumann", neumann_prototype, \
                                                float(str_list[4]), (int(str_list[5]), int(str_list[6])), float(str_list[7])))
                            print(u_boundaries[len(u_boundaries) - 1])
                        else:
                            raise RuntimeError('Boundary Format Error')
                    elif str_list[0] == "v":
                        if str_list[3] == "Dirichlet" and len(str_list) == 5:
                            v_boundaries.append(DirichletBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(v_mesh == int(str_list[1])), "Dirichlet", dirichlet_prototype, \
                                                    float(str_list[4])))
                            print(v_boundaries[len(v_boundaries) - 1])
                        elif str_list[3] == "Neumann" and len(str_list) == 8:
                            v_boundaries.append(NeumannBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(v_mesh == int(str_list[1])), "Neumann", neumann_prototype, \
                                                float(str_list[4]), (int(str_list[5]), int(str_list[6])), float(str_list[7])))
                            print(v_boundaries[len(v_boundaries) - 1])
                        else:
                            raise RuntimeError('Boundary Format Error')
                    elif str_list[0] == "p":
                        if str_list[3] == "Dirichlet" and len(str_list) == 5:
                            p_boundaries.append(DirichletBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(p_mesh == int(str_list[1])), "Dirichlet", dirichlet_prototype, \
                                                    float(str_list[4])))
                        elif str_list[3] == "Neumann" and len(str_list) == 8:
                            p_boundaries.append(NeumannBoundary(int(str_list[1]), str_list[0] + "_" + str_list[2], \
                                                np.where(p_mesh == int(str_list[1])), "Neumann", neumann_prototype, \
                                                float(str_list[4]), (int(str_list[5]), int(str_list[6])), float(str_list[7])))
                        else:
                            raise RuntimeError('Boundary Format Error')
                        print(p_boundaries[len(p_boundaries) - 1])
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
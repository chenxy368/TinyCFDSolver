# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 14:17:28 2022

@author: HP
"""
import numpy as np
from utils.boundary import BoundaryCondition

class ImplicitBoundary2D(BoundaryCondition):
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: tuple, shape: tuple,
                 boundary_type = "Implicit Base 2D", boundary_parameters_list = []):
        super(ImplicitBoundary2D, self).__init__(boundary_id, boundary_name, boundary_domain, boundary_type, boundary_parameters_list)
        self.shape = shape
        self.RHS_domain = ()
        
    def flatten_domain(self, boundary_domain: tuple, shape: tuple, offset: tuple):
        node_num = len(boundary_domain[0])
        RHS_domain = np.zeros([node_num], dtype = int)

        for i in range(node_num):
            RHS_domain[i] = (boundary_domain[1][i] + offset[1]) * shape[0] + (boundary_domain[0][i] + offset[0])
        
        return (RHS_domain,)
    
    def modify_RHS_domain(self, new_RHS: np.ndarray):
        node_num = len(self.RHS_domain[0])

        index_list = []
        for i in range(node_num):
            index = np.where(new_RHS == self.RHS_domain[0][i])
            if index[0].shape[0] == 0:
                continue
            
            index_list.append(index[0][0])
        
        self.RHS_domain = (np.array(index_list, dtype=int),)
    
class FirstOrderCondition2D(ImplicitBoundary2D):
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: list, shape: tuple, 
                 boundary_parameters_list = []):
        super(FirstOrderCondition2D, self).__init__(boundary_id, boundary_name, boundary_domain, shape,
                                                    "Implicit First Order 2D", boundary_parameters_list)
        self.bias_RHS, self.bias_domain, self.diff, self.offset = self.parse_parameters(boundary_parameters_list)
        self.RHS_domain = self.flatten_domain(boundary_domain, shape, self.offset)
        
    def parse_parameters(self, boundary_parameters_list):
        return boundary_parameters_list[3], boundary_parameters_list[4], boundary_parameters_list[0], (int(boundary_parameters_list[1]), 
               int(boundary_parameters_list[2]))
    
    def process(self, obj):
        obj[self.domain] = self.bias_domain
        return obj
    
    def process_RHS(self, RHS):
        RHS[self.RHS_domain] += self.bias_RHS
        return RHS
    
    def modify_matrix(self, A):
        node_num = len(self.domain[0])
        A_slice = np.zeros([node_num], dtype = int)
        for i in range(node_num):
            A_slice[i] = (self.domain[0][i] + self.offset[0]) * self.shape[1] + self.domain[1][i] + self.offset[1]
        
        A_slice = (A_slice, A_slice)
        
        A[A_slice] += self.diff
        
        return A

    def get_params(self):
        return [self.diff, self.offset, self.bias_RHS, self.bias_domain]

class ImplicitBoundary1D(BoundaryCondition):
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: int,
                 boundary_type = "Implicit Base 1D", boundary_parameters_list = []):
        super(ImplicitBoundary1D, self).__init__(boundary_id, boundary_name, boundary_domain, boundary_type, boundary_parameters_list)
        self.RHS_domain = None
    
    def set_RHS_domain(self, offset: int, boundary_domain:int):
        return boundary_domain + offset
        
    def modify_RHS_domain(self, new_RHS: tuple):
        index = np.where(new_RHS[0] == self.RHS_domain)
        if index[0].shape[0] == 0:
            return
        
        self.RHS_domain = index[0][0]
        
class FirstOrderCondition1D(ImplicitBoundary1D):
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: int, boundary_parameters_list = []):
        super(FirstOrderCondition1D, self).__init__(boundary_id, boundary_name, boundary_domain,
                                                    "Implicit First Order 1D", boundary_parameters_list)
        self.bias_RHS, self.bias_domain, self.diff, self.offset = self.parse_parameters(boundary_parameters_list)
        self.RHS_domain = self.set_RHS_domain(boundary_domain, self.offset)
        
    def parse_parameters(self, boundary_parameters_list):
        return boundary_parameters_list[2], boundary_parameters_list[3], boundary_parameters_list[0], int(boundary_parameters_list[1])
    
    def process(self, obj):
        obj[self.domain] = self.bias_domain
        return obj
    
    def process_RHS(self, RHS):
        RHS[self.RHS_domain] += self.bias_RHS
        return RHS
    
    def modify_matrix(self, A):
        A[self.domain + self.offset, self.domain + self.offset] += self.diff
        
        return A    
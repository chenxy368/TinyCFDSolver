# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 03:12:39 2022

@author: HP
"""
import numpy as np
from utils.boundary import ConstCondition, LinearCondition, \
                           LinearCombinationCondition, \
                           NQuantityLinearCombinationCondition, \
                           LinearSpacialCondition

boundary_classes = {"Const": ConstCondition,
                    "Linear": LinearCondition, 
                    "LinearCombination": LinearCombinationCondition, 
                    "NQuantityLinearCombination": NQuantityLinearCombinationCondition,
                    "LinearSpacial": LinearSpacialCondition}


class GridLoader():
    def __init__(self, root: str):
        self.root = root
        
    def parse_grid_info(self, filename: str, domain_dict: dict, boundaries_dict: dict, mesh_dict: dict):
        START = 0
        PARSING_METHOD = 1
        PARSING_PARAMETERS = 2
        PARSING_DOMAIN_ID = 3
        PARSING_BOUNDARIES = 4
        
        state = START

        parameters_dict = {}
        
        with open(self.root + "/" + filename) as file:
            for line in file:
                str_list = line.split()
                if state == START:
                    if str_list[0] == "METHOD":
                        state = PARSING_METHOD
                        continue
                elif state == PARSING_METHOD:
                    if str_list[0] == "PARAMETER":
                        state = PARSING_PARAMETERS
                        continue
                    method_info = str_list
                elif state == PARSING_PARAMETERS:
                    if str_list[0] == "DOMAIN":
                        state = PARSING_DOMAIN_ID
                        continue
                    parameters_dict = self.parse_parameter(str_list, parameters_dict)
                elif state == PARSING_DOMAIN_ID:
                    if str_list[0] == "BOUNDARY":
                        state = PARSING_BOUNDARIES
                        continue
                    domain_dict = self.parse_domain(str_list, domain_dict)
                elif state == PARSING_BOUNDARIES:
                    boundaries_dict = self.parse_boundary(str_list, boundaries_dict, mesh_dict, parameters_dict)                          
                else:
                    raise RuntimeError('PARSING ERROR')
              
        if state != PARSING_BOUNDARIES:  
            raise RuntimeError('INCOMLETE GRID INFORMATION')
        
        return  method_info, parameters_dict, domain_dict, boundaries_dict
    
    def parse_domain(self, str_list: list, domain_dict: dict):
        assert len(str_list) >= 2
        
        domain_dict[str_list[0]] = int(str_list[1])
        return domain_dict
    
    def parse_parameter(self, str_list: list, parameters_dict: dict):
        assert len(str_list) == 2
        
        if str_list[0] not in parameters_dict:
            parameters_dict[str_list[0]] = str_list[1]
        return parameters_dict
    
    def parse_boundary(self, str_list: list, boundaries_dict: dict, mesh_dict: dict, parameters_dict: dict):
        if str_list[0] in boundaries_dict:
            boundaries_dict[str_list[0]].append(self.create_boundary(str_list, mesh_dict[str_list[0]], parameters_dict))
        else:
            raise RuntimeError('Unknown Physical Quantity')
        return boundaries_dict 
    
    def create_boundary(self, boundary_information: list, mesh: np.ndarray, parameters_dict: dict):
        assert len(boundary_information) > 3
        
        boundary_name = boundary_information[0] + "_" + boundary_information[2]
        boundary_id = int(boundary_information[1])
        boundary_domain = np.where(mesh == int(boundary_information[1]))
        boundary_params = boundary_information[4:]

        if boundary_information[3] not in boundary_classes:
            raise RuntimeError('Boundary Type Error')
            
        for index in range(len(boundary_params)):
            for key in parameters_dict:
                if key in boundary_params[index]:
                    boundary_params[index] = boundary_params[index].replace(key, parameters_dict[key])

            boundary_params[index] = float(eval(boundary_params[index]))

        new_boundary = boundary_classes[boundary_information[3]](boundary_id, boundary_name, boundary_domain, boundary_params)
        
        print(new_boundary)
        
        return new_boundary
    
    def load_grid(self, domain_dict: dict, mesh_boundary_dict: dict):
        for key in domain_dict:
            domain_dict[key] = np.flipud(np.loadtxt(self.root + "/" + domain_dict[key], delimiter=",", dtype = int)).transpose()
        
        mesh_dict = {}
        boundaries_dict ={}
        for key in mesh_boundary_dict:
            boundaries_list = mesh_boundary_dict[key]
            for boundary in boundaries_list:
                mesh_dict[boundary] = domain_dict[key]
                boundaries_dict[boundary] = []
        
        shape_dict = {}
        interior_ID_dict = {}
        for key in domain_dict:
            shape_dict[key] = domain_dict[key].shape
            interior_ID_dict[key] = -1
        
        method_info, parameters_dict, interior_ID_dict, boundaries_dict = self.parse_grid_info("grid_information.txt", interior_ID_dict, boundaries_dict, mesh_dict)
        
        for key in parameters_dict:
            parameters_dict[key] = float(parameters_dict[key])
        
        for key in interior_ID_dict:
            domain_dict[key + "_exterior"] = np.where(domain_dict[key] != interior_ID_dict[key])
            domain_dict[key] = np.where(domain_dict[key] == interior_ID_dict[key])
        
        return method_info, (shape_dict, parameters_dict, domain_dict, boundaries_dict)

class BlendSFVGridLoader2D(GridLoader):
    def __init__(self, root: str):
        super(BlendSFVGridLoader2D, self).__init__(root)
        
    def load_grid(self, domain_dict: dict, mesh_boundary_dict: dict, base_mesh_keyword: str):
        base_mesh = np.flipud(np.loadtxt(self.root + "/" + domain_dict[base_mesh_keyword], delimiter=",", dtype = int)).transpose()
        for key in domain_dict:
            domain_dict[key] = np.flipud(np.loadtxt(self.root + "/" + domain_dict[key], delimiter=",", dtype = int)).transpose()
        
        mesh_dict = {}
        boundaries_dict ={}
        for key in mesh_boundary_dict:
            boundaries_list = mesh_boundary_dict[key]
            for boundary in boundaries_list:
                mesh_dict[boundary] = domain_dict[key]
                boundaries_dict[boundary] = []
        
        shape_dict = {}
        interior_ID_dict = {}
        for key in domain_dict:
            shape_dict[key] = domain_dict[key].shape
            interior_ID_dict[key] = -1
        
        method_info, parameters_dict, interior_ID_dict, boundaries_dict = self.parse_grid_info("grid_information.txt", interior_ID_dict, boundaries_dict, mesh_dict)
        base_mesh_id = interior_ID_dict[base_mesh_keyword]
        for key in parameters_dict:
            parameters_dict[key] = float(parameters_dict[key])
        
        for key in interior_ID_dict:
            domain_dict[key + "_exterior"] = np.where(domain_dict[key] != interior_ID_dict[key])
            domain_dict[key] = np.where(domain_dict[key] == interior_ID_dict[key])
        
        base_mesh_out_id = base_mesh_id - 1
        base_exterior = np.where(base_mesh != base_mesh_id)
        base_mesh[base_exterior] = base_mesh_out_id
        
        wx_minus_mesh = np.zeros_like(base_mesh)
        wx_plus_mesh = np.zeros_like(base_mesh)
        wy_minus_mesh = np.zeros_like(base_mesh)
        wy_plus_mesh = np.zeros_like(base_mesh)
        wx_minus_mesh[...] = base_mesh[...]
        wx_plus_mesh[...] = base_mesh[...]
        wy_minus_mesh[...] = base_mesh[...]
        wy_plus_mesh[...] = base_mesh[...]
        base_shape = base_mesh.shape

        for i in range(len(base_exterior[0])):
            if base_exterior[0][i] + 1 < base_shape[0] and base_mesh[base_exterior[0][i] + 1, base_exterior[1][i]] == base_mesh_id:
                wx_minus_mesh[base_exterior[0][i] + 1, base_exterior[1][i]] = base_mesh_out_id
            if base_exterior[0][i] - 1 >= 0 and base_mesh[base_exterior[0][i] - 1, base_exterior[1][i]] == base_mesh_id:
                wx_plus_mesh[base_exterior[0][i] - 1, base_exterior[1][i]] = base_mesh_out_id
            if base_exterior[1][i] + 1 < base_shape[1] and base_mesh[base_exterior[0][i], base_exterior[1][i] + 1] == base_mesh_id:
                wy_minus_mesh[base_exterior[0][i] + 1, base_exterior[1][i] + 1] = base_mesh_out_id
            if base_exterior[1][i] - 1 >= 0 and base_mesh[base_exterior[0][i], base_exterior[1][i] - 1] == base_mesh_id:
                wy_plus_mesh[base_exterior[0][i], base_exterior[1][i] - 1] = base_mesh_out_id

        domain_dict["wx_minus"] = np.where(wx_minus_mesh == base_mesh_id)
        domain_dict["wx_plus"] = np.where(wx_plus_mesh == base_mesh_id)
        domain_dict["wy_minus"] = np.where(wy_minus_mesh == base_mesh_id)
        domain_dict["wy_plus"] = np.where(wy_plus_mesh == base_mesh_id)
        
        return method_info, (shape_dict, parameters_dict, domain_dict, boundaries_dict)
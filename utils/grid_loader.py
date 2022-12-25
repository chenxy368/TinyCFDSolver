# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 03:12:39 2022

@author: HP
"""
import numpy as np
from utils.boundary import ConstCondition, LinearCondition, \
                           LinearCombinationCondition, \
                           TwoQuantityLinearCombinationCondition

boundary_classes = {"Const": ConstCondition,
                    "Linear": LinearCondition, 
                    "LinearCombination": LinearCombinationCondition, 
                    "TwoQuantityLinearCombination": TwoQuantityLinearCombinationCondition}


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

class BlendSchemeGridLoader2D(GridLoader):
    def __init__(self, root: str):
        super(BlendSchemeGridLoader2D, self).__init__(root)
        
    def load_grid(self):
        mesh = np.flipud(np.loadtxt(self.root + "/mesh.csv", delimiter=",", dtype = int)).transpose()
        blend_mesh = np.flipud(np.loadtxt(self.root + "/blend_scheme_mesh.csv", delimiter=",", dtype = int)).transpose()
        
        domain_dict = {
            "mesh": -1,
            "blend_mesh": -1
        }
        mesh_dict = {
            "u": mesh, 
            "v": mesh,
            "psi": mesh,
            "w_v_psi": mesh,
            "w_u_psi": mesh,
            "wx_plus": blend_mesh, 
            "wy_plus": blend_mesh,
            "wx_minus": blend_mesh,
            "wy_minus": blend_mesh
        }
        
        boundaries_dict = {
            "u": [], 
            "v": [],
            "psi": [],
            "w_v_psi": [],
            "w_u_psi": [],
            "wx_plus": [], 
            "wy_plus": [],
            "wx_minus": [],
            "wy_minus": []
        }
        
        method_info, parameters_dict, domain_dict, boundaries_dict = self.parse_grid_info("grid_information.txt", domain_dict, boundaries_dict, mesh_dict)
        
        domain_dict["mesh_exterior"] = np.where(mesh != domain_dict["mesh"]) 
        domain_dict["mesh"] = np.where(mesh == domain_dict["mesh"]) 
        domain_dict["blend_mesh"] = np.where(blend_mesh == domain_dict["blend_mesh"])
        
        for key in parameters_dict:
            parameters_dict[key] = float(parameters_dict[key])
        
        return method_info, (mesh.shape, parameters_dict, domain_dict, boundaries_dict)
    
class StaggeredGridLoader(GridLoader):
    def __init__(self, root: str):
        super(StaggeredGridLoader, self).__init__(root)
        
    def load_grid(self):
        u_mesh = np.flipud(np.loadtxt(self.root + "/u_mesh.csv", delimiter=",", dtype = int)).transpose()
        v_mesh = np.flipud(np.loadtxt(self.root + "/v_mesh.csv", delimiter=",", dtype = int)).transpose()
        p_mesh = np.flipud(np.loadtxt(self.root + "/p_mesh.csv", delimiter=",", dtype = int)).transpose()
        
        domain_dict = {
            "u": -1,
            "v": -1,
            "p": -1
        }
        boundaries_dict ={
            "u": [],
            "v": [],
            "p": []
        }
        mesh_dict = {
            "u": u_mesh,
            "v": v_mesh,
            "p": p_mesh
        }

        method_info, parameters_dict, domain_dict, boundaries_dict = self.parse_grid_info("grid_information.txt", domain_dict, boundaries_dict, mesh_dict)
        
        domain_dict["u_exterior"] = np.where(u_mesh != domain_dict["u"]) 
        domain_dict["v_exterior"] = np.where(v_mesh != domain_dict["v"])
        domain_dict["p_exterior"] = np.where(p_mesh != domain_dict["p"]) 
        domain_dict["u"] = np.where(u_mesh == domain_dict["u"]) 
        domain_dict["v"] = np.where(v_mesh == domain_dict["v"])
        domain_dict["p"] = np.where(p_mesh == domain_dict["p"]) 
      
        for key in parameters_dict:
            parameters_dict[key] = float(parameters_dict[key])  
      
        return method_info, ((u_mesh.shape, v_mesh.shape, p_mesh.shape), parameters_dict, domain_dict, boundaries_dict)
               
    
   
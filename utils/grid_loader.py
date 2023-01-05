# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 03:12:39 2022

Define grid loader in this file

@author: Xinyang Chen
"""
import numpy as np
import re

from utils.boundary import ConstCondition, LinearCondition, \
                           LinearCombinationCondition, \
                           NQuantityLinearCombinationCondition, \
                           LinearSpacialCondition
                           
from utils.implicit_boundary import FirstOrderCondition2D, FirstOrderCondition1D

"""
Dictionary of implicit and explicit boundary classes:
    Key: Type name
    Value: Class init function
"""
boundary_classes = {"Const": ConstCondition,
                    "Linear": LinearCondition, 
                    "LinearCombination": LinearCombinationCondition, 
                    "NQuantityLinearCombination": NQuantityLinearCombinationCondition,
                    "LinearSpacial": LinearSpacialCondition}
implcit_boundry_classes = {"FirstOrder2D": FirstOrderCondition2D,
                           "FirstOrder1D": FirstOrderCondition1D}


class GridLoader():
    """Grid Loader Class
        
        Load grid matrices and parse the grid information
        
    Attributes:
        root: path to the root of project
    """    
    def __init__(self, root: str):
        self.root = root
        
    def parse_grid_info(self, filename: str, domain_dict: dict, boundaries_dict: dict, mesh_dict: dict):
        """ Parse the grid information
        
        Run a finite state machine to parse the grid information file
        An example grid information file:
            METHOD
            FracStep PointJacobi 0.9
            PARAMETER
            dx 0.1
            dy 0.1
            dt 0.001
            density 1000/1
            kinematic_viscosity 0.001
            inflow_speed 0.05
            DOMAIN
            u_mesh -2
            v_mesh -2
            p_mesh -1
            BOUNDARY
            u 0 Wall_Dirichlet Const 0
            ...
            u 7 Outlet_Ghost_Neumann Linear 1 0 -1 0
            v 0 Wall_Dirichlet Const 0
            ...
            v 7 Outlet_Ghost_Neumann Linear 1 0 -1 0
            p1 0 Wall_Dirichlet Const 0
            ...
            p1 7 Outlet_Dirichlet Const 0
            p2 0 Wall_Dirichlet Const 0
            ...
            p2 7 Outlet_Dirichlet Const 0
        
        Args:
            filename: file name of grid information txt
            domain_dict: a dictionary of physical quantities, keys are the physical quantities names. 
                         values are mesh ID arrays
            boundaries_dict: a dictionary of physical quantities and boundaries, keys are the physical quantities names, 
                              values are boundary classes(init as empty lists)
            mesh_dict: a dictionary of boundaries and grids, keys are physical quantities, values are mesh ID arrays  
        Return:
            method_info: the name of the method
            parameters_dict: a dictionary of parameters' name and parameters, keys are parameters' name, 
                             values are parameters
            domain_dict: a dictionary of physical quantites and mesh ID arrays, keys are physical quantites, 
                         values are mesh ID arrays
            boundaries_dict: a dictionary of physical quantitise and boundary classes, keys are physical quantities,
                             values are boundary classes
        """
        
        # State ID
        START = 0
        PARSING_METHOD = 1
        PARSING_PARAMETERS = 2
        PARSING_DOMAIN_ID = 3
        PARSING_BOUNDARIES = 4
        
        state = START
        parameters_dict = {}
        
        # Parse the grid information with a finite state machine
        # START -> PARSING_METHOD -> PARSING_PARAMETERS -> PARSING_DOMAIN_ID -> PARSING_BOUNDARIES
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
        
        # Assert reaching the end
        if state != PARSING_BOUNDARIES:  
            raise RuntimeError('INCOMLETE GRID INFORMATION')
        
        return  method_info, parameters_dict, domain_dict, boundaries_dict
    
    def parse_domain(self, str_list: list, domain_dict: dict):
        """ 
        Get interior ID
        """
        assert len(str_list) >= 2
        
        domain_dict[str_list[0]] = int(str_list[1])
        return domain_dict
    
    def parse_parameter(self, str_list: list, parameters_dict: dict):
        """ 
        Get parameter
        """
        assert len(str_list) == 2
        
        if str_list[0] not in parameters_dict:
            parameters_dict[str_list[0]] = str_list[1]

        return parameters_dict
    
    def parse_boundary(self, str_list: list, boundaries_dict: dict, mesh_dict: dict, parameters_dict: dict):
        """ Create boundary with boundary information
            
        An example input: 
            u 0 Inlet_Dirichlet Const Inflow_velocity
            -> boundaries_dict["u"](boundary group of u).append(self.create_boundary(
                ["0", "Inlet_Dirichlet", "Const", "Inflow_velocity"], 
                mesh_dict["u"], parameters_dict))
        """
        if str_list[0] in boundaries_dict:
            boundaries_dict[str_list[0]].append(self.create_boundary(str_list, mesh_dict[str_list[0]], parameters_dict))
        else:
            raise RuntimeError('Unknown Physical Quantity')
        return boundaries_dict 
    
    def create_boundary(self, boundary_information: list, mesh: np.ndarray, parameters_dict: dict):
        """ Create boundary with boundary information
            
        An example input: 
            boundary_information: ["0", "Inlet_Dirichlet", "Const", "Inflow_velocity"]
            mesh: u's mesh array
            parameters_dict = {
                ...
                "Inflow_velocity": "0.05",
                ...
            }

            Create a "Const" type boundary with id = 0, name = "Inlet_Dirichlet", domain = np.where(mesh == 0)
            and information list = [0.05]
        Args:
            boundary_information: see discription
            mesh: see discription
            parameters_dict: see discription
        Return:
            a boundary object
        """
        
        assert len(boundary_information) > 3
        
        boundary_name = boundary_information[0] + "_" + boundary_information[2]
        boundary_id = int(boundary_information[1])
        boundary_domain = np.where(mesh == int(boundary_information[1]))
        boundary_params = boundary_information[4:]
        
        # Undefined Boundary
        if boundary_information[3] not in boundary_classes and boundary_information[3] not in implcit_boundry_classes:
            raise RuntimeError('Boundary Type Error')
        
        # Substitude parameter with num
        for index in range(len(boundary_params)):
            for key in parameters_dict:
                if key in boundary_params[index] and re.match('^[-0-9+*/.()]+$', boundary_params[index].replace(key, parameters_dict[key])):
                    boundary_params[index] = boundary_params[index].replace(key, parameters_dict[key])

            boundary_params[index] = float(eval(boundary_params[index]))
        
        # Create boundary
        if boundary_information[3] in boundary_classes:
            new_boundary = boundary_classes[boundary_information[3]](boundary_id, boundary_name, boundary_domain, boundary_params)
        else:
            new_boundary = implcit_boundry_classes[boundary_information[3]](boundary_id, boundary_name, boundary_domain, mesh.shape, boundary_params)
        
        print(new_boundary)
        
        return new_boundary
    
    def load_grid(self, domain_dict: dict, mesh_boundary_dict: dict):
        """ Load grids arrays and generate boundary classes
        
        For an example input:
            The keys of domain_dict are the name of the mesh arrays.
            The values of domain_dict are the filename of the girds, which is a csv file storing ID. With ID, mesh arrays 
            can specify where is the interior and where is the boundaries. ID also link boundary position with boundary 
            types.
            domain_dict = {
                "u_mesh": "u_mesh.csv",
                "v_mesh": "v_mesh.csv",
                "p_mesh": "p_mesh.csv"
            }
            The keys of mesh_boundary_dict are the name of the mesh arrays.
            The values of mesh_boundary_dict are the boundary groups. Boundaries are correponding to different physical
            quantities. Since boundary classes need to have a field about where is the boundary position, mesh_boundary_dict
            help them get the position with given grid arrays. In brief, call np.where to get the slice in arrays with unique
            ID of each boundary.
            mesh_boundary_dict = {
                "u_mesh": ["u"],
                "v_mesh": ["v"],
                "p_mesh": ["p1, p2"]
            }
            
            Process:
                1. Load grids
                2. Create mesh_dict and boundaries_dict, e.g.:
                    mesh_dict = {
                        "u": u_mesh array,
                        "v": v_mesh array,
                        "p1": p_mesh array,
                        "p2": p_mesh array
                    }
                    boundaries_dict = {
                        "u": [],
                        "v": [],
                        "p1": [],
                        "p2": []
                    }
                3. Create shape_dict and interior_ID_dict, e.g.:
                    shape_dict = {
                        "u_mesh": (..., ...),
                        "v_mesh": (..., ...),
                        "p_mesh": (..., ...)
                    }
                    interior_ID_dict = {
                        "u_mesh": -1,
                        "v_mesh": -1,
                        "p_mesh": -1
                    }
                4. Parse grid information:
                    An example grid information file:
                        METHOD
                        FracStep PointJacobi 0.9
                        PARAMETER
                        dx 0.1
                        dy 0.1
                        dt 0.001
                        density 1000/1
                        kinematic_viscosity 0.001
                        inflow_speed 0.05
                        DOMAIN
                        u_mesh -2
                        v_mesh -2
                        p_mesh -1
                        BOUNDARY
                        u 0 Wall_Dirichlet Const 0
                        ...
                        u 7 Outlet_Ghost_Neumann Linear 1 0 -1 0
                        v 0 Wall_Dirichlet Const 0
                        ...
                        v 7 Outlet_Ghost_Neumann Linear 1 0 -1 0
                        p1 0 Wall_Dirichlet Const 0
                        ...
                        p1 7 Outlet_Dirichlet Const 0
                        p2 0 Wall_Dirichlet Const 0
                        ...
                        p2 7 Outlet_Dirichlet Const 0
                     method_info = ["FracStep", "PointJacobi", "0.9"]
                     parameters_dict = {
                         "dx": "0.1",
                         "dy": "0.1",
                         "dt": "0.001",
                         "density": "1000/1",
                         "kinematic_viscosity": "0.001", 
                         "inflow_speed": "0.05"
                     }
                     interior_ID_dict = {
                         u_mesh: -2,
                         v_mesh: -2,
                         p_mesh: -1
                     }
                     boundaries_dict == {
                         "u": [list of boundary classes],
                         "v": [list of boundary classes],
                         "p1": [list of boundary classes],
                         "p2": [list of boundary classes]
                     }
                5. Eval all parameters:
                     parameters_dict = {
                         "dx": 0.1
                         "dy": 0.1
                         "dt": 0.001
                         "density": 1000
                         "kinematic_viscosity": 0.001
                         "inflow_speed": 0.05
                     }
                6. Create return mesh_dict and domain_dict:
                    mesh_dict = {
                        "u_mesh": u_mesh array,
                        "v_mesh": v_mesh array,
                        "p_mesh": p_mesh array
                    }
                    domain_dict = {
                        "u_mesh": u_mesh array interior slice list,
                        "v_mesh": v_mesh array interior slice list,
                        "p_mesh": p_mesh array interior slice list,
                        "u_mesh_exterior": u_mesh array exterior slice list,
                        "v_mesh_exterior": v_mesh array exterior slice list,
                        "p_mesh_exterior": p_mesh array exterior slice list
                    }
        
        Args:
            domain_dict: see discription
            mesh_boundary_dict: see discription
        Return:
            method_info: see discription
            parameters_dict: see discription
            domain_dict: see discription
            boundaries_dict: see discription
            mesh_dict: see discription
        """
        # Load grids with csv files
        for key in domain_dict:
            domain_dict[key] = np.flipud(np.loadtxt(self.root + "/" + domain_dict[key], delimiter=",", dtype = int)).transpose()
        
        # Keys: name of physical quantites, Values: grid array
        mesh_dict = {}
        # Keys: name of physical quantites, Values: boundary classes
        boundaries_dict ={}
        for key in mesh_boundary_dict:
            boundaries_list = mesh_boundary_dict[key]
            for boundary in boundaries_list:
                mesh_dict[boundary] = domain_dict[key]
                boundaries_dict[boundary] = []
        
        # Shape of each grid and init interior_ID dictionary
        # Keys: name of physical quantites, Values: shape tuple
        shape_dict = {}
        # Keys: name of physical quantites, Values: interior ID
        interior_ID_dict = {}
        for key in domain_dict:
            shape_dict[key] = domain_dict[key].shape
            interior_ID_dict[key] = -1
        
        # Parse grid information
        method_info, parameters_dict, interior_ID_dict, boundaries_dict = self.parse_grid_info("grid_information.txt", interior_ID_dict, boundaries_dict, mesh_dict)
        
        # Eval all the parameters
        for key in parameters_dict:
            parameters_dict[key] = float(eval(parameters_dict[key]))
        
        # Save grid dictionary and change domain_dict's values to the slice of grids corresponding to interior and exterior
        mesh_dict.clear()
        for key in interior_ID_dict:
            mesh_dict[key] = domain_dict[key]
            domain_dict[key + "_exterior"] = np.where(domain_dict[key] != interior_ID_dict[key])
            domain_dict[key] = np.where(domain_dict[key] == interior_ID_dict[key])
        
        return method_info, (shape_dict, parameters_dict, domain_dict, boundaries_dict), mesh_dict

class BlendSFVGridLoader2D(GridLoader):
    """Grid Loader Class of blended scheme streamfunction vortex method
        
        Load grid matrices and parse the grid information, generate the blended grid and set its boundary
        An example of blended grid (1: exterior, 0: interior):
            1 1 1 1 1 1 1                1 1 1 1 1 1 1
            1 0 0 0 0 0 1                1 1 1 1 1 1 1
            1 0 0 0 0 0 1 ------------>  1 0 0 0 0 0 1 (wy_plus blend calculation. Since cannot get j+2 at interior
            1 0 0 0 0 0 1                1 0 0 0 0 0 1 nodes near the top with third order accuracy in the method,
            1 1 1 1 1 1 1                1 1 1 1 1 1 1 mark them as exteriror when calculate blended factor) 
        
    Attributes:
        root: path to the root of project
    """    
    def __init__(self, root: str):
        super(BlendSFVGridLoader2D, self).__init__(root)
        
    def load_grid(self, domain_dict: dict, mesh_boundary_dict: dict, base_mesh_keyword: str):
        """ Load grids arrays and generate boundary classes, generate blended grids
      
        Args:
            domain_dict: same as base class
            mesh_boundary_dict: same as base class
        Return:
            method_info: same as base class
            parameters_dict: same as base class
            domain_dict: same as base class
            boundaries_dict: same as base class
            mesh_dict: same as base class
        """
        # Load grid
        method_info, info_tuple, mesh_dict = super(BlendSFVGridLoader2D, self).load_grid(domain_dict, mesh_boundary_dict)
        
        # Generate blended grid
        shape_dict = info_tuple[0]
        parameters_dict = info_tuple[1]
        domain_dict = info_tuple[2]
        boundaries_dict = info_tuple[3]
        
        base_mesh = mesh_dict[base_mesh_keyword]
        base_mesh_id = base_mesh[domain_dict[base_mesh_keyword][0][0], domain_dict[base_mesh_keyword][1][0]]
        
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
        
        return method_info, (shape_dict, parameters_dict, domain_dict, boundaries_dict), mesh_dict
    
class FracStepGridLoader2D(GridLoader):
    """Grid Loader Class of fractional step method
        
        Load grid matrices and parse the grid information, generate the staggered grid and set its boundary
        An example of staggered grid (u, v and p grids are staggered):
            ---U-----U-----U---
            V  P  V  P  V  P  V
            ---U-----U-----U---
            V  P  V  P  V  P  V
            ---U-----U-----U---
            V  P  V  P  V  P  V
            ---U-----U-----U---
        
        An example of generation (2: exterior, 0: interior, 3, 4, 5, 6: gost node, 7: fix boundary):
            DOMAIN
            mesh 0
            Ghost 3 4 5 6
            Expand_u 2 3 5 
            Translate u 3 7
            Translate u 5 7
 
        
            5 5 5 5 5 5 5 5 5                 5 5 5 5 5 5 5 5                   7 7 7 7 7 7 7 7
            6 0 0 0 0 0 0 0 4                 6 0 0 0 0 0 0 4                   6 0 0 0 0 0 0 4
            6 0 0 0 0 0 0 0 4                 6 0 0 0 0 0 0 4                   6 0 0 0 0 0 0 4
            6 0 0 3 3 3 0 0 4                 6 0 3 3 3 3 0 4                   6 0 7 7 7 7 0 4
            6 0 0 4 2 6 0 0 4 --------------> 6 0 4 2 2 6 0 4 ----------------> 6 0 4 2 2 6 0 4 
            6 0 0 5 5 5 0 0 4   generate u    6 0 5 5 5 5 0 4     substitude    6 0 7 7 7 7 0 4  
            6 0 0 0 0 0 0 0 4                 6 0 0 0 0 0 0 4                   6 0 0 0 0 0 0 4
            6 0 0 0 0 0 0 0 4                 6 0 0 0 0 0 0 4                   6 0 0 0 0 0 0 4
            3 3 3 3 3 3 3 3 3                 3 3 3 3 3 3 3 3                   3 3 3 3 3 3 3 3          
        (full grid, p's grid)
    Attributes:
        root: path to the root of project
    """    
    def __init__(self, root: str):
        super(FracStepGridLoader2D, self).__init__(root)
        self.staggered_dict = {}
        self.ghost_list = []
        self.u_expand_list = []
        self.v_expand_list = []
    
    def parse_domain(self, str_list: list, domain_dict: dict):
        """ 
        Get interior ID
        """
        assert len(str_list) >= 2
        # Skip lines for staggered grids generation
        if str_list[0] != "Translate" and str_list[0] != "Ghost" and str_list[0] != "Expand_u" \
            and str_list[0] != "Expand_v":                  
            domain_dict[str_list[0]] = int(str_list[1])
            
        return domain_dict
    
    def load_grid(self, domain_dict: dict, mesh_boundary_dict: dict):
        """ Load grids arrays and generate boundary classes, generate u's and v's grid based on full stagger grid
      
        Args:
            domain_dict: see base class
            mesh_boundary_dict: see base class
        Return:
            method_info: see base class
            parameters_dict: see base class
            domain_dict: see base class
            boundaries_dict: see base class
            mesh_dict: see base class
        """
        if len(domain_dict) == 0:
            raise RuntimeError("Empty Domain Dictionary")

        # Load grids with csv files
        for key in domain_dict:
            domain_dict[key] = np.flipud(np.loadtxt(self.root + "/" + domain_dict[key], delimiter=",", dtype = int)).transpose()
            p_mesh = domain_dict[key]

        # Create u and v girds
        u_mesh = np.zeros([p_mesh.shape[0]-1, p_mesh.shape[1]], dtype = int)
        v_mesh = np.zeros([p_mesh.shape[0], p_mesh.shape[1]-1], dtype = int)
        
        # Parse staggered grid information
        SEARCH = 0
        PARSING_DOMAIN = 1
        state = SEARCH
        with open(self.root + "/" + "grid_information.txt") as file:
            for line in file:
                str_list = line.split()
                if state == SEARCH and str_list[0] == "DOMAIN":
                    state = PARSING_DOMAIN
                elif state == PARSING_DOMAIN:
                    if str_list[0] == "Translate":                  
                        self.staggered_dict[(str_list[1], int(str_list[2]))] = int(str_list[3])
                    elif str_list[0] == "Ghost":
                        self.ghost_list = str_list[1:]
                        for i in range(len(self.ghost_list)):
                            self.ghost_list[i] = int(self.ghost_list[i])
                    elif str_list[0] == "Expand_u":
                        self.u_expand_list = str_list[1:]
                        for i in range(len(self.u_expand_list)):
                            self.u_expand_list[i] = int(self.u_expand_list[i])
                    elif str_list[0] == "Expand_v":
                        self.v_expand_list = str_list[1:]
                        for i in range(len(self.v_expand_list)):
                            self.v_expand_list[i] = int(self.v_expand_list[i])
                    elif str_list[0] == "BOUNDARY":
                        break
                        
        # Generate u gird
        for j in range(p_mesh.shape[1]):
            for i in range(p_mesh.shape[0]-1):
                if p_mesh[i, j] == p_mesh[i+1, j]:
                    u_mesh[i, j] = p_mesh[i, j]
                    continue
                
                if p_mesh[i, j] in self.u_expand_list and p_mesh[i+1, j] in self.u_expand_list:
                    if self.u_expand_list.index(p_mesh[i, j]) < self.u_expand_list.index(p_mesh[i+1, j]):
                        u_mesh[i, j] = p_mesh[i, j]
                    else:
                        u_mesh[i, j] = p_mesh[i+1, j]
                elif p_mesh[i, j] in self.u_expand_list:
                    u_mesh[i, j] = p_mesh[i, j]
                elif p_mesh[i+1, j] in self.u_expand_list:
                    u_mesh[i, j] = p_mesh[i+1, j]
                elif p_mesh[i, j] in self.ghost_list:
                    u_mesh[i, j] = p_mesh[i, j]
                elif p_mesh[i+1, j] in self.ghost_list:
                    u_mesh[i, j] = p_mesh[i+1, j]
                else:
                    u_mesh[i, j] = p_mesh[i, j]
        
        # Generate v grid
        for i in range(p_mesh.shape[0]):
            for j in range(p_mesh.shape[1]-1):    
                if p_mesh[i, j] == p_mesh[i, j+1]:
                    v_mesh[i, j] = p_mesh[i, j]
                    continue
                
                if p_mesh[i, j] in self.v_expand_list and p_mesh[i, j+1] in self.v_expand_list:
                    if self.v_expand_list.index(p_mesh[i, j]) < self.v_expand_list.index(p_mesh[i, j+1]):
                        v_mesh[i, j] = p_mesh[i, j]
                    else:
                        v_mesh[i, j] = p_mesh[i, j+1]
                elif p_mesh[i, j] in self.v_expand_list:
                    v_mesh[i, j] = p_mesh[i, j]
                elif p_mesh[i, j+1] in self.v_expand_list:
                    v_mesh[i, j] = p_mesh[i, j+1]
                elif p_mesh[i, j] in self.ghost_list:
                    v_mesh[i, j] = p_mesh[i, j]
                elif p_mesh[i, j+1] in self.ghost_list:
                    v_mesh[i, j] = p_mesh[i, j+1]
                else:
                    v_mesh[i, j] = p_mesh[i, j]
        
        # Substitude ghost node with fixed boundary
        for key in self.staggered_dict:
            if key[0] == "u":
                u_mesh[np.where(u_mesh == key[1])] = self.staggered_dict[key]
            elif key[0] == "v":     
                v_mesh[np.where(v_mesh == key[1])] = self.staggered_dict[key]

        # Keys: name of physical quantites, Values: grid array
        mesh_dict = {
            "u": u_mesh,    
            "v": v_mesh,    
            "p": p_mesh
        }

        # Keys: name of physical quantites, Values: boundary classes
        boundaries_dict = {}
        for key in mesh_boundary_dict:
            boundaries_list = mesh_boundary_dict[key]
            for boundary in boundaries_list:
                boundaries_dict[boundary] = []
        

        # Keys: name of physical quantites, Values: shape tuple
        shape_dict = {
            "u": u_mesh.shape,    
            "v": v_mesh.shape,    
            "p": p_mesh.shape  
        }
        
        # Keys: name of physical quantites, Values: interior ID
        interior_ID_dict = {}
        for key in domain_dict:
            interior_ID_dict[key] = -1
        
        # Parse grid information
        method_info, parameters_dict, interior_ID_dict, boundaries_dict = self.parse_grid_info("grid_information.txt", interior_ID_dict, boundaries_dict, mesh_dict)
        
        # Eval all the parameters
        for key in parameters_dict:
            parameters_dict[key] = float(eval(parameters_dict[key]))
        
        # Save grid dictionary and change domain_dict's values to the slice of grids corresponding to interior and exterior
        domain_dict = {
            "u": u_mesh,    
            "v": v_mesh,    
            "p": p_mesh
        }

        # Get interior ID
        interior_ID = -1
        for key in interior_ID_dict:
            interior_ID = interior_ID_dict[key]

        # Get slices
        domain_dict["u_exterior"] = np.where(domain_dict["u"] != interior_ID)
        domain_dict["u"] = np.where(domain_dict["u"] == interior_ID)
        domain_dict["v_exterior"] = np.where(domain_dict["v"] != interior_ID)
        domain_dict["v"] = np.where(domain_dict["v"] == interior_ID)
        domain_dict["p_exterior"] = np.where(domain_dict["p"] != interior_ID)
        domain_dict["p"] = np.where(domain_dict["p"] == interior_ID)
        
        return method_info, (shape_dict, parameters_dict, domain_dict, boundaries_dict), mesh_dict
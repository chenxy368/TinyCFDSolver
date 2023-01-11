# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 14:17:28 2022

@author: Xinyang Chen
"""
import numpy as np
from utils.boundary import BoundaryCondition

class ImplicitBoundary2D(BoundaryCondition):
    """Implicit 2D boundary Condition Base Class
    
    A base class for defining implicit boundary conditions. Require overwrite the process, parse_parameters
    and helper methods in the child class. Provide flatten_domain and modify_RHS_domain functions for implicit
    condition setting.
    
    Attributes:
        shape: the id of the boundary, same as the id representing it in the grid file 
        RHS_domain: slice for right hand side processing
    """
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: tuple, shape: tuple,
                 boundary_type = "Implicit Base 2D", boundary_parameters_list = []):
        super(ImplicitBoundary2D, self).__init__(boundary_id, boundary_name, boundary_domain, boundary_type, boundary_parameters_list)
        self.shape = shape
        self.RHS_domain = ()
        
    def flatten_domain(self, boundary_domain: tuple, shape: tuple, offset: tuple):
        """ Flatten the dimain to get slice for right hand side

        Args:
            boundary_domain: boundary position
            shape: shape of the mesh
            offset: position offset
        Return:
            a processing slice of right hand side
        """
        node_num = len(boundary_domain[0])
        RHS_domain = np.zeros([node_num], dtype = int)

        for i in range(node_num):
            RHS_domain[i] = (boundary_domain[1][i] + offset[1]) * shape[0] + (boundary_domain[0][i] + offset[0])
        
        return (RHS_domain,)
    
    def modify_RHS_domain(self, RHS_domain: list, new_RHS: np.ndarray):
        """ Change the right hand side slice with new_RHS position array

        Args:
            RHS_domain: right hand side slice
            new_RHS: new_RHS position array
        Return:
            a processing slice of right hand side
        """
        node_num = len(RHS_domain[0])

        index_list = []
        for i in range(node_num):
            index = np.where(new_RHS == RHS_domain[0][i])
            if index[0].shape[0] == 0:
                continue
            
            index_list.append(index[0][0])
        
        RHS_domain = (np.array(index_list, dtype=int),)
        return RHS_domain
    
class ImplicitLinearCondition2D(ImplicitBoundary2D):
    """Implicit 2D Linear boundary Condition
    
    A prototype for implicit boundary with formula: obj[x, y] = a1*obj[x+offset1, y] + a2*obj[x, y+offset2]  + const
    Process right hand side: b[domain_x] += bias_x and b[domain_y] += bias_y
    
    Attributes:
        diff_x, diff_y: for linear system Ax = b, difference between original matrix and processed matrix
        coefficient_x, coefficient_y: a1 and a2 
        offset_x, offset_y: offset1 and offset2
        bias_RHS_x, bias_RHS_y: bias_x and bias_y
        bias_domain: const
        RHS_domain_x, RHS_domain_y: domain_x and domain_y
    """
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: list, shape: tuple, 
                 boundary_parameters_list = []):
        super(ImplicitLinearCondition2D, self).__init__(boundary_id, boundary_name, boundary_domain, shape,
                                                    "Implicit Linear 2D", boundary_parameters_list)
        self.diff_x, self.coefficient_x, self.offset_x, self.bias_RHS_x, self.diff_y, self.coefficient_y, \
            self.offset_y, self.bias_RHS_y, self.bias_domain = self.parse_parameters(boundary_parameters_list)
        if not isinstance(self.offset_x, str):
            self.RHS_domain_x = self.flatten_domain(boundary_domain, shape, (self.offset_x, 0))
        else:
            self.RHS_domain_x = (np.array([]),)
            self.offset_x = 0
        
        if not isinstance(self.offset_y, str):
            self.RHS_domain_y = self.flatten_domain(boundary_domain, shape, (0, self.offset_y))
        else:
            self.RHS_domain_y = (np.array([]),)
            self.offset_y = 0
        
    def modify_RHS_domain(self, new_RHS: np.ndarray):
        """ 
        modify RHS domain along two direction
        """
        self.RHS_domain_x = super(ImplicitLinearCondition2D, self).modify_RHS_domain(self.RHS_domain_x, new_RHS)
        self.RHS_domain_y = super(ImplicitLinearCondition2D, self).modify_RHS_domain(self.RHS_domain_y, new_RHS)
         
    def parse_parameters(self, boundary_parameters_list):
        diff_x = 0
        coefficient_x = 0
        offset_x = "_"
        bias_RHS_x = 0
        # Check whether process x direction
        if not isinstance(boundary_parameters_list[2], str):
            diff_x = boundary_parameters_list[0]
            coefficient_x = boundary_parameters_list[1]
            offset_x = int(boundary_parameters_list[2])
            bias_RHS_x = boundary_parameters_list[3]
        diff_y = 0
        coefficient_y = 0
        offset_y = "_"
        bias_RHS_y = 0
        # Check whether process y direction
        if not isinstance(boundary_parameters_list[6], str):
            diff_y = boundary_parameters_list[4]
            coefficient_y = boundary_parameters_list[5]
            offset_y = int(boundary_parameters_list[6])
            bias_RHS_y = boundary_parameters_list[7]
        
        return diff_x, coefficient_x, offset_x, bias_RHS_x, diff_y, coefficient_y, offset_y, bias_RHS_y, boundary_parameters_list[8]
    
    def process(self, obj):
        obj[self.domain] = self.coefficient_x * obj[self.domain[0] + self.offset_x, self.domain[1]] \
            + self.coefficient_y * obj[self.domain[0], self.domain[1] + self.offset_y] + self.bias_domain
        return obj
    
    def process_RHS(self, RHS):
        """ 
        process RHS vector
        """
        RHS[self.RHS_domain_x] += self.bias_RHS_x
        RHS[self.RHS_domain_y] += self.bias_RHS_y
        return RHS
    
    def modify_matrix(self, A):
        """ 
        modify A in Ax = b
        """
        if not isinstance(self.offset_x, str):
            node_num = len(self.domain[0])
            A_slice = np.zeros([node_num], dtype = int)
            for i in range(node_num):
                A_slice[i] = (self.domain[0][i] + self.offset_x) * self.shape[1] + self.domain[1][i]
        
            A_slice = (A_slice, A_slice)
        
            A[A_slice] += self.diff_x
            
        if not isinstance(self.offset_y, str):
            node_num = len(self.domain[0])
            A_slice = np.zeros([node_num], dtype = int)
            for i in range(node_num):
                A_slice[i] = self.domain[0][i] * self.shape[1] + self.domain[1][i] + self.offset_y
        
            A_slice = (A_slice, A_slice)
        
            A[A_slice] += self.diff_y 
        
        return A
    
    @staticmethod
    def helper(self):
        format_str = "ImplicitLinearCondition format: Group ID Name ImplicitLinear2D diff_x coefficient_x offset_x bias_RHS_x diff_y coefficient_y offset_y bias_RHS_y bias"
        formula_str = "Domain: y[i, j] = C0 * y[i+O0, j] + C1 * y[i, j+O1] + Bias, Right Hand Side: b[i] += bias"
        return format_str, formula_str  

    def get_params_x(self):
        return [self.diff_x, self.coefficient_x, self.offset_x, self.bias_RHS_x, self.bias_domain]
            
    def get_params_y(self):
        return [self.diff_y, self.coefficient_y, self.offset_y, self.bias_RHS_y, self.bias_domain]

class ImplicitLinearCondition2DTime(ImplicitLinearCondition2D):
    """Implicit 2D Linear time boundary Condition
    
    A prototype for implicit boundary with formula: obj[x, y] = a1*obj[x+offset1, y] + a2*obj[x, y+offset2]  + const(t)
    Process right hand side: b[domain_x] += bias_x(t) and b[domain_y] += bias_y(t)
    The const(t), bias_x(t) and bias_y(t) is a function of timestep.
    
    Attributes:
        time_index: current check points index
        time_key: check points
        dt: unit timestep length
        curr_bias_RHS_x, curr_bias_RHS_y: bias_x(time_key[time_index]) and bias_y(time_key[time_index])
        curr_bias_domian: const(time_key[time_index])
    """
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: list, shape: tuple, 
                 boundary_parameters_list = []):
        super(ImplicitLinearCondition2DTime, self).__init__(boundary_id, boundary_name, boundary_domain, shape,
                                                        boundary_parameters_list)
        self.type = "Implicit Linear 2D Time"
        self.time_index = 0
        self.time_key = np.array(list(self.bias_domain.keys()), dtype = float)
        self.dt = 0
        if self.bias_RHS_x != {}:
            self.curr_bias_RHS_x = self.bias_RHS_x[self.time_key[0]]
        else:
            self.curr_bias_RHS_x = 0
        if self.bias_RHS_y != {}:
            self.curr_bias_RHS_y = self.bias_RHS_y[self.time_key[0]]
        else:
            self.curr_bias_RHS_y = 0
        self.curr_bias_domian = self.bias_domain[self.time_key[0]]
        self.reset_curr_params(0)
    
    def set_dt(self, dt: float):
        """ 
        set time interval
        """
        self.dt = dt
    
    def reset_curr_params(self, t: int):
        """ 
        set curr_bias_RHS_x, curr_bias_RHS_y, and curr_bias_domian based on timestep
        """
        time = t * self.dt
        if self.time_index == self.time_key.shape[0]-1 or time < self.time_key[self.time_index+1]:
            return

        self.time_index += 1
        curr_key = self.time_key[self.time_index]
        self.curr_bias_RHS_x = 0
        self.curr_bias_RHS_y = 0
       
        if self.bias_RHS_x != {}:
            self.curr_bias_RHS_x = self.bias_RHS_x[curr_key]
        if self.bias_RHS_y != {}:
            self.curr_bias_RHS_y = self.bias_RHS_y[curr_key]

        self.curr_bias_domian = self.bias_domain[curr_key]

    def process(self, obj):
        obj[self.domain] = self.coefficient_x * obj[self.domain[0] + self.offset_x, self.domain[1]] \
            + self.coefficient_y * obj[self.domain[0], self.domain[1] + self.offset_y] + self.curr_bias_domian
        return obj
    
    def process_RHS(self, RHS):
        RHS[self.RHS_domain_x] += self.curr_bias_RHS_x
        RHS[self.RHS_domain_y] += self.curr_bias_RHS_y
        return RHS
    
    def parse_parameters(self, boundary_parameters_list):
        time_point = boundary_parameters_list[9]
        
        diff_x = 0
        coefficient_x = 0
        offset_x = "_"
        bias_RHS_x = {}
        if not isinstance(boundary_parameters_list[2], str):
            diff_x = boundary_parameters_list[0]
            coefficient_x = boundary_parameters_list[1]
            offset_x = int(boundary_parameters_list[2])
            bias_RHS_x = dict(zip(time_point, boundary_parameters_list[3]))

        diff_y = 0
        coefficient_y = 0
        offset_y = "_"
        bias_RHS_y = {}
        if not isinstance(boundary_parameters_list[6], str):
            diff_y = boundary_parameters_list[4]
            coefficient_y = boundary_parameters_list[5]
            offset_y = int(boundary_parameters_list[6])
            bias_RHS_y = dict(zip(time_point, boundary_parameters_list[7]))
            
        return diff_x, coefficient_x, offset_x, bias_RHS_x, diff_y, coefficient_y, offset_y, bias_RHS_y, \
            dict(zip(time_point, boundary_parameters_list[8]))

    @staticmethod
    def helper(self):
        format_str = "ImplicitLinearCondition format: Group ID Name ImplicitLinear2DTime diff_x coefficient_x offset_x bias_RHS_x diff_y coefficient_y offset_y bias_RHS_y bias"
        formula_str = "Domain: y[i, j] = C0 * y[i+O0, j] + C1 * y[i, j+O1] + Bias, Right Hand Side: b[i] += bias"
        return format_str, formula_str

class ImplicitBoundary1D(BoundaryCondition):
    """Implicit 1D boundary Condition Base Class
    
    A base class for defining implicit boundary conditions. Require overwrite the process, parse_parameters
    and helper methods in the child class. Provide flatten_domain and modify_RHS_domain functions for implicit
    condition setting.
    
    Attributes:
        RHS_domain: slice for right hand side processing
    """
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
        
class ImplicitLinearCondition1D(ImplicitBoundary1D):
    """Implicit 1D Linear boundary Condition
    
    A prototype for implicit boundary with formula: obj[x] = a*obj[x+offset] + const
    Process right hand side: b[domain] += bias
    
    Attributes:
        diff: for linear system Ax = b, difference between original matrix and processed matrix
        coefficient: a in the formula
        offset: offset in the formula
        bias_RHS: bias in the formula
        bias_domain: const in the formula
        RHS_domain: domain in the formula
    """
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: int, boundary_parameters_list = []):
        super(ImplicitLinearCondition1D, self).__init__(boundary_id, boundary_name, boundary_domain,
                                                    "Implicit Linear 1D", boundary_parameters_list)
        self.bias_RHS, self.bias_domain, self.diff, self.coefficient, self.offset = self.parse_parameters(boundary_parameters_list)
        self.RHS_domain = self.set_RHS_domain(boundary_domain, self.offset)
        
    def parse_parameters(self, boundary_parameters_list):
        return boundary_parameters_list[3], boundary_parameters_list[4], boundary_parameters_list[0], \
            boundary_parameters_list[1], int(boundary_parameters_list[2])
    
    def process(self, obj):
        obj[self.domain] = self.coefficient * obj[self.domain + self.offset] + self.bias_domain
        return obj
    
    def process_RHS(self, RHS):
        RHS[self.RHS_domain] += self.bias_RHS
        return RHS
    
    def modify_matrix(self, A):
        A[self.domain + self.offset, self.domain + self.offset] += self.diff
        
        return A    
    
    @staticmethod
    def helper(self):
        format_str = "This is used for ADI to process 2D conditions. Do not call directly"
        formula_str = "Domain: y[i] = C * y[i+O0] + Bias, Right Hand Side: b[i] += bias"
        return format_str, formula_str
    
class ImplicitLinearCondition1DTime(ImplicitLinearCondition1D):
    """Implicit 1D Linear time boundary Condition
    
    A prototype for implicit boundary with formula: obj[x, y] = a1*obj[x+offset1, y] + const(t)
    Process right hand side: b[domain_x] += bias(t)
    The const(t), and bias(t) is a function of timestep.
    
    Attributes:
        time_index: current check points index
        time_key: check points
        dt: unit timestep length
        curr_bias_RHS: bias(time_key[time_index])
        curr_bias_domian: const(time_key[time_index])
    """
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: int, boundary_parameters_list = []):
        super(ImplicitLinearCondition1DTime, self).__init__(boundary_id, boundary_name, boundary_domain, 
                                                        boundary_parameters_list)
        self.type = "Implicit Linear 1D Time"
        self.time_index = 0
        self.time_key = np.array(list(self.bias_domain.keys()), dtype = float)
        self.dt = 0
        self.curr_bias_RHS = self.bias_RHS[self.time_key[0]]
        self.curr_bias_domian = self.bias_domain[self.time_key[0]]
        self.reset_curr_params(0)
        
        
    def set_dt(self, dt: float):
        self.dt = dt
    
    def reset_curr_params(self, t: int):
        time = t * self.dt
        if self.time_index == self.time_key.shape[0]-1 or time < self.time_key[self.time_index+1]:
            return 
        self.time_index += 1
        curr_key = self.time_key[self.time_index]

        self.curr_bias_RHS = self.bias_RHS[curr_key]
        self.curr_bias_domian = self.bias_domain[curr_key]
    def process(self, obj):
        obj[self.domain] = self.coefficient * obj[self.domain + self.offset] + self.curr_bias_domian
        return obj
    
    def process_RHS(self, RHS):
        RHS[self.RHS_domain] += self.curr_bias_RHS
        return RHS
    
    def parse_parameters(self, boundary_parameters_list):
        if isinstance(boundary_parameters_list[3], dict) and isinstance(boundary_parameters_list[4], dict):
            return boundary_parameters_list[3], boundary_parameters_list[4], boundary_parameters_list[0], \
                boundary_parameters_list[1], int(boundary_parameters_list[2])
        
        time_point = boundary_parameters_list[5]
        diff = boundary_parameters_list[0]
        coefficient = boundary_parameters_list[1]
        offset = int(boundary_parameters_list[2])

        return dict(zip(time_point, boundary_parameters_list[3])), dict(zip(time_point, boundary_parameters_list[4])), \
            diff, coefficient, offset
            
    @staticmethod
    def helper(self):
        format_str = "This is used for ADI to process 2D conditions. Do not call directly"
        formula_str = "Domain: y[i] = C * y[i+O0] + Bias, Right Hand Side: b[i] += bias"
        return format_str, formula_str    
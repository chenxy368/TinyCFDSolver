# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:02:30 2022

Define boundary classes in this file

@author: Xinyang Chen
"""
import numpy as np

class BoundaryCondition():
    """Boundary Condition Base Class
    
    A base class for defining boundary conditions. Require overwrite the process, parse_parameters
    and helper methods in the child class.
    
    Attributes:
        id: the id of the boundary, same as the id representing it in the grid file 
        name: name of this boundary
        domain: a slice define the position in grid
        type: the type of the boundary
    """
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: tuple, boundary_type = "Base", 
                 boundary_parameters_list = []):
        """ Inits with boundary_id, boundary_name, boundary_domain and boundary_type. Also provide a string list
            to inits parameters for this boundary type
        """
        self.id = boundary_id
        self.name = boundary_name
        self.domain = boundary_domain
        self.type = boundary_type
        self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj):
        """ Process the input object with boundary condition
       
            In this method, for different boundary conditions, they need to use the slice to get required domain
            and set the value with its formula and parameters.
       
        Args:
            obj: object to change value at the boundary area
        Return:
            processed obj
        """
        raise RuntimeWarning("Warning: implement the process method")
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        """ Parse the parameters from txt file about boundary condition setting
       
            For different boundary conditions, they may have different required parameters. The loader sends
            a string list containing all required information for parameters. Write a parser to parse and 
            set parameters.
       
        Args:
            boundary_parameters_list: a string list contains boundary condition parameters
        """
        raise RuntimeWarning("Warning: implement the parse_parameters method")
        for parameters in boundary_parameters_list:
            print(parameters + " ")
    
    def set_domain(self, domain: list):
        self.domain = domain
    
    def get_id(self):
        return self.id
    
    def get_name(self):
        return self.name
    
    def get_type(self):
        return self.type
    
    def get_domain(self):
        return self.domain
    
    @staticmethod
    def helper(self):
        """ Help message
       
        Args:
            format_str: required format to define boundary condition in grid_information txt file
            formula_str: the formula of boundary condition
        """
        raise RuntimeWarning("Warning: implement the static helper method")
        format_str = "Define boundary declaration format in grid_information.txt"
        formula_str = "Define boundary process formula in process method"
        return format_str, formula_str  
    
    def __str__(self):
        return "ID: " + str(self.id) + ", Name: " +  self.name + ", Type: " + self.type
    
class ConstCondition(BoundaryCondition):
    """Const Boundary Condition Class
    
        A prototype for boundary with formula: y[domian] = const
    
    Attributes:
        id: the id of the boundary, same as the id representing it in the grid file 
        name: name of this boundary
        domain: a slice define the position in grid
        type: the type of the boundary
        bias: const number on the right hand side
    """
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: tuple, boundary_parameters_list:list):       
        super(ConstCondition, self).__init__(boundary_id, boundary_name, boundary_domain,  "Const", boundary_parameters_list)
        self.bias = self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj):
        obj[self.domain] = self.bias
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        assert len(boundary_parameters_list) == 1
        
        return boundary_parameters_list[0]
        
    def set_bias(self, bias:float):
        self.bias = bias

    @staticmethod
    def helper(self):
        format_str = "ConstCondition format: Group ID Name Const Bias"
        formula_str = "y[i, j] = Bias"
        return format_str, formula_str    

    def __str__(self):
        return super(ConstCondition, self).__str__() + ", Formula: y = " + str(self.bias) 

class LinearCondition(BoundaryCondition):
    """Linear Boundary Condition Class
    
        A prototype for boundary with formula: y[domian] = a*y[domain+offset] + const
    
    Attributes:
        id: the id of the boundary, same as the id representing it in the grid file 
        name: name of this boundary
        domain: a slice define the position in grid
        type: the type of the boundary
        bias: const number on the right hand side
        offset: define which node is used to compute boundary value, e.g (1, 0) use the right node next to boundary
        coefficient: a in the formula
    """
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: tuple, boundary_parameters_list: list):
        super(LinearCondition, self).__init__(boundary_id, boundary_name, boundary_domain, "Linear", boundary_parameters_list)
        self.bias, self.offset, self.coefficient = self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj):
        obj[self.domain] = self.coefficient * obj[self.domain[0] + self.offset[0], self.domain[1] + self.offset[1]] + self.bias
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        assert len(boundary_parameters_list) == 4 
        
        return boundary_parameters_list[3], (int(boundary_parameters_list[1]), int(boundary_parameters_list[2])), \
               boundary_parameters_list[0]
    
    def set_bias(self, bias: float):
        self.bias = bias
        
    def set_coefficient(self, coefficient: float):
        self.coefficient = coefficient
    
    @staticmethod
    def helper(self):
        format_str = "LinearCondition format: Group ID Name Linear C O0 O1 Bias"
        formula_str = "y[i, j] = C0 * y[i+O0, j+O1] + Bias"
        return format_str, formula_str  
    
    def __str__(self):
        output_str = super(LinearCondition, self).__str__() + ", Formula: y[i, j] = " + str(self.coefficient) +  "·y[i"
        if self.offset[0] >= 0:
            output_str += "+"
        output_str += str(self.offset[0]) + ", j"
        if self.offset[1] >= 0:
            output_str += "+"
        output_str += str(self.offset[1]) + "]"
        if self.bias >= 0:
            output_str += "+"
        output_str += str(self.bias)
        return output_str
    
class LinearCombinationCondition(BoundaryCondition):
    """Linear Combination Boundary Condition Class
    
        A prototype for boundary with formula: y[domian] = a0*x[domain+offset0] + a1*x[domain+offset1] + ... + const
        Note that x can be y or other variables.
    
    Attributes:
        id: the id of the boundary, same as the id representing it in the grid file 
        name: name of this boundary
        domain: a slice define the position in grid
        type: the type of the boundary
        bias: const number on the right hand side
        offset: define which node is used to compute boundary value, e.g (1, 0) use the right node next to boundary
        coefficient: a0, a1... in the formula
    """
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: tuple, boundary_parameters_list: list):     
        super(LinearCombinationCondition, self).__init__(boundary_id, boundary_name, boundary_domain, "LinearCombination", boundary_parameters_list)
        self.bias, self.quant_coefficient, self.offsets, self.coefficients = self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj, obj0):
        obj[self.domain] = self.bias
        item_num = len(self.offsets)
        for i in range(item_num):
            obj[self.domain] += self.coefficients[i] * obj0[self.domain[0] + self.offsets[i][0], self.domain[1] + self.offsets[i][1]]
        obj[self.domain] *= self.quant_coefficient
        
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        assert len(boundary_parameters_list) > 0 and len(boundary_parameters_list) == 3 + int(boundary_parameters_list[0]) * 3
        
        offsets = []
        coefficients = []
        quant_coefficient = boundary_parameters_list[1]
        for i in range(int(boundary_parameters_list[0])):
            coefficients.append(boundary_parameters_list[i * 3 + 2])
            offsets.append((int(boundary_parameters_list[i * 3 + 3]), int(boundary_parameters_list[i * 3 + 4])))
                
        return boundary_parameters_list[len(boundary_parameters_list)-1], quant_coefficient, offsets, coefficients
    
    def set_bias(self, bias: float):
        self.bias = bias
        
    def set_coefficient(self, coefficient: list):
        self.coefficient = coefficient
    
    @staticmethod
    def helper(self):
        format_str = "LinearCombinationCondition format: Group ID Name LinearCombination #Term Term0_C0 Term0_C1 Term0_O0 Term0_O1 Term1_C0 ... TermN_O1 Bias"
        formula_str = "y[i, j] = C00 * x0[i+O00, j+O01] * C01 + C10 * x1[i+O10, j+O11] * C11 + ... + Bias"
        return format_str, formula_str
    
    def __str__(self):
        output_str = super(LinearCombinationCondition, self).__str__() + \
                    ", Formula: y[i, j] = " + str(self.quant_coefficient) + "·("
        for i in range(len(self.offsets)):
            if i != 0 and self.coefficients[i] >= 0:
                output_str += "+"
            output_str += str(self.coefficients[i]) + "·x" + str(i) + "[i"
            if self.offsets[i][0] >= 0:
                output_str += "+"
            output_str += str(self.offsets[i][0]) + ", j"
            if self.offsets[i][1] >= 0:
                output_str += "+" 
            output_str += str(self.offsets[i][1]) + "])"
            
        if self.bias >= 0:
            output_str += "+" 
        output_str += str(self.bias)
        return  output_str
    
class NQuantityLinearCombinationCondition(BoundaryCondition):
    """Multiple Variables Linear Combination Boundary Condition Class
    
        A prototype for boundary with formula: y[domian] = a00*x0[domain+offset00] + ... + a0n*x1[domain+offset0n] + 
        a10*x1[domain+offset10] + ... a1m*x1[domain+offset1m] + ... + const
        Note that x0, x1, ... can be y or other variables.
    
    Attributes:
        id: the id of the boundary, same as the id representing it in the grid file 
        name: name of this boundary
        domain: a slice define the position in grid
        type: the type of the boundary
        bias: const number on the right hand side
        offset: define which node is used to compute boundary value, e.g (1, 0) use the right node next to boundary
                arrange in shape [((O000, O001), (O010, O011), ....), ((O100, O101), (O110, O111), ...), ...]
        coefficient: a00, a01... in the formula, arrange in shape [(a00, a01, ....), (a10, a11, ...), ...]
    """
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: tuple, boundary_parameters_list: list):        
        super(NQuantityLinearCombinationCondition, self).__init__(boundary_id, boundary_name, boundary_domain, "NQuantityLinearCombination", boundary_parameters_list)
        self.bias, self.quant_coefficients, self.offsets, self.coefficients = self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj, objs: tuple):
        obj[self.domain] = self.bias
        quant_num = len(self.offsets)
        for i in range(quant_num):
            curr_offsets = self.offsets[i]
            curr_coefficients = self.coefficients[i]
            curr_res = np.zeros_like(obj)
            item_num = len(curr_offsets)
            for j in range(item_num):
                curr_res[self.domain] += curr_coefficients[j] * objs[i][self.domain[0] + curr_offsets[j][0], \
                                    self.domain[1] + curr_offsets[j][1]]
            curr_res = curr_res * self.quant_coefficients[i]
            obj[self.domain] += curr_res[self.domain]
        
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        offsets = []
        quant_coefficients = []
        coefficients = []
        start = 1
        quant_num = int(boundary_parameters_list[0])
        for i in range(quant_num):
            item_num = int(boundary_parameters_list[start])
            quant_coefficients.append(boundary_parameters_list[start + 1])
            curr_coefficients = []
            curr_offsets = []
            for j in range(item_num):
                curr_coefficients.append(boundary_parameters_list[start + j * 3 + 2])
                curr_offsets.append((int(boundary_parameters_list[start + j * 3 + 3]), int(boundary_parameters_list[start + j * 3 + 4])))
            coefficients.append(curr_coefficients)
            offsets.append(curr_offsets)
            start += item_num * 3 + 2
            
        return boundary_parameters_list[len(boundary_parameters_list)-1], quant_coefficients, offsets, coefficients
    
    def set_bias(self, bias:float):
        self.bias = bias
        
    def set_coefficient(self, coefficient: list):
        self.coefficient = coefficient
    
    @staticmethod
    def helper(self):
        format_str = "NQuantityLinearCombinationCondition format: Group ID Name LinearCombination #Quantity #Term \
                    Quandtity0_C Quantity0_Term0_C Quantity0_Term0_O0 Quantity0_Term0_O1 ... Quantity1_Term0_C ... \
                    QuantityN_TermN_O1 Bias"
        formula_str = "y[i, j] = C0 * (C00 * x00[i+O000, j+O001] + ... +) + ... (... + Cnm * xnm[i+Onm0, j+Onm1] + ...) \
                        + ... + Bias"
        return format_str, formula_str
    
    def __str__(self):
        output_str = super(NQuantityLinearCombinationCondition, self).__str__() + ", Formula: y[i, j] = "

        for i in range(len(self.offsets)):
            item_num = len(self.offsets[i])
            if i != 0 and self.quant_coefficients[i] >= 0:
                output_str += "+"
            output_str += str(self.quant_coefficients[i]) + "·("
            for j in range(item_num):
                if j != 0 and self.coefficients[i][j] >= 0:
                    output_str += "+"
                output_str += str(self.coefficients[i][j]) + "·x" + str(i) + str(j) + "[i"
                if self.offsets[i][j][0] >= 0:
                    output_str += "+"
                output_str += str(self.offsets[i][j][0]) + ", j"
                if self.offsets[i][j][1] >= 0:
                    output_str += "+" 
                output_str += str(self.offsets[i][j][1]) + "]"
            output_str += ")"
        if self.bias >= 0:
            output_str += "+" 
        output_str += str(self.bias)
        return  output_str
    
class LinearSpacialCondition(BoundaryCondition):
    """Linear Spacial Boundary Condition Class
    
        A prototype for boundary with formula: y[i, j] = a0*i + a1*j + const 
    
    Attributes:
        id: the id of the boundary, same as the id representing it in the grid file 
        name: name of this boundary
        domain: a slice define the position in grid
        type: the type of the boundary
        bias: const number on the right hand side
        coefficient: a0, a1 in the formula
    """
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: tuple, boundary_parameters_list: list):        
        super(LinearSpacialCondition, self).__init__(boundary_id, boundary_name, boundary_domain, "LinearSpacial", boundary_parameters_list)
        self.bias, self.coefficients = self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj):
        obj[self.domain] = self.coefficients[0] * self.domain[0] + self.coefficients[1] * self.domain[1] + self.bias
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        return boundary_parameters_list[2], (boundary_parameters_list[0], boundary_parameters_list[1])
    
    def set_bias(self, bias:float):
        self.bias = bias
        
    def set_coefficient(self, coefficient: tuple):
        self.coefficients = coefficient
     
    @staticmethod
    def helper(self):
        format_str = "LinearSpacialCondition format: Group ID Name LinearCombination C0 C1 bias"
        formula_str = "y[i, j] = C0 * i + C1 * j + Bias"
        return format_str, formula_str
    
    def __str__(self):
        return super(LinearSpacialCondition, self).__str__() + \
            ", Formula: y[i, j] = " + str(self.coefficients[0]) + "* i + " + str(self.coefficients[1]) + "* j + " + str(self.bias)
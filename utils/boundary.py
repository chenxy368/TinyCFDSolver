# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:02:30 2022

@author: HP
"""

class BoundaryCondition():
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: list, boundary_type = "Base", 
                 boundary_parameters_list = []):
        self.id = boundary_id
        self.name = boundary_name
        self.domain = boundary_domain
        self.type = boundary_type
        self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj):
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        print("Warning: Do not call boundary condition base")
    
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
    
    def __str__(self):
        return "ID: " + str(self.id) + ", Name: " +  self.name + ", Type: " + self.type
    
class DirichletBoundary(BoundaryCondition):
    def __init__(self, boundary_id: int, boundary_name: str, boundary_domain: list, boundary_parameters_list:list):       
        super(DirichletBoundary, self).__init__(boundary_id, boundary_name, boundary_domain,  "Dirichlet", boundary_parameters_list)
        self.bias = self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj):
        obj[self.domain] = self.bias
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        assert len(boundary_parameters_list) == 1
        
        return float(boundary_parameters_list[0])
        
    def set_bias(self, bias:float):
        self.bias = bias

class NeumannBoundary(BoundaryCondition):
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: list, boundary_parameters_list: list):
        super(NeumannBoundary, self).__init__(boundary_id, boundary_name, boundary_domain, "Neumann", boundary_parameters_list)
        self.bias, self.offset, self.coefficient = self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj):
        obj[self.domain] = self.coefficient * obj[self.domain[0] + self.offset[0], self.domain[1] + self.offset[1]] + self.bias
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        assert len(boundary_parameters_list) == 4 
        
        return float(boundary_parameters_list[0]), (int(boundary_parameters_list[1]), int(boundary_parameters_list[2])), \
               float(boundary_parameters_list[3])
    
    def set_bias(self, bias: float):
        self.bias = bias
        
    def set_coefficient(self, coefficient: float):
        self.coefficient = coefficient
    
class LinearCombinationCondition(BoundaryCondition):
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: list, boundary_parameters_list: list):     
        super(LinearCombinationCondition, self).__init__(boundary_id, boundary_name, boundary_domain, "LinearCombination", boundary_parameters_list)
        self.bias, self.offsets, self.coefficients = self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj, obj0):
        obj[self.domain] = self.bias
        item_num = len(self.offsets)
        for i in range(item_num):
            obj[self.domain] += self.coefficients[i] * obj0[self.domain[0] + self.offsets[i][0], self.domain[1] + self.offsets[i][1]]
        
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        assert len(boundary_parameters_list) > 0 and len(boundary_parameters_list) == 2 + int(boundary_parameters_list[0]) * 3
        
        offsets = []
        coefficients = []
        for i in range(int(boundary_parameters_list[0])):
            coefficients.append(float(boundary_parameters_list[i * 3 + 1]))
            offsets.append((int(boundary_parameters_list[i * 3 + 2]), int(boundary_parameters_list[i * 3 + 3])))
                
        return float(boundary_parameters_list[len(boundary_parameters_list)-1]), offsets, coefficients
    
    def set_bias(self, bias: float):
        self.bias = bias
        
    def set_coefficient(self, coefficient: list):
        self.coefficient = coefficient
    
class TwoQuantityLinearCombinationCondition(BoundaryCondition):
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: list, boundary_parameters_list: list):        
        super(TwoQuantityLinearCombinationCondition, self).__init__(boundary_id, boundary_name, boundary_domain, "TwoQuantityLinearCombination", boundary_parameters_list)
        self.bias, self.offsets1, self.coefficients1, self.offsets2, self.coefficients2 = self.parse_parameters(boundary_parameters_list)
        
    def process(self, obj, obj1, obj2):
        obj[self.domain] = self.bias
        item_num = len(self.offsets1)
        for i in range(item_num):
            obj[self.domain] += self.coefficients1[i] * obj1[self.domain[0] + self.offsets1[i][0], self.domain[1] + self.offsets1[i][1]]
        item_num = len(self.offsets2)
        for i in range(item_num):
            obj[self.domain] += self.coefficients2[i] * obj2[self.domain[0] + self.offsets2[i][0], self.domain[1] + self.offsets2[i][1]]
            
        return obj
    
    def parse_parameters(self, boundary_parameters_list):
        offsets1 = []
        coefficients1 = []
        for i in range(int(boundary_parameters_list[0])):
            coefficients1.append(float(boundary_parameters_list[i * 3 + 1]))
            offsets1.append((int(boundary_parameters_list[i * 3 + 2]), int(boundary_parameters_list[i * 3 + 3])))
            
        offsets2 = []
        coefficients2 = []
        for i in range(int(boundary_parameters_list[3 * int(boundary_parameters_list[0]) + 1])):
            coefficients2.append(float(boundary_parameters_list[i * 3 + 3 * int(boundary_parameters_list[0]) + 2]))
            offsets2.append((int(boundary_parameters_list[i * 3 + 3 * int(boundary_parameters_list[0]) + 3]), int(boundary_parameters_list[i * 3 + 3 * int(boundary_parameters_list[0]) + 4])))

        return float(boundary_parameters_list[len(boundary_parameters_list)-1]), offsets1, coefficients1, offsets2, coefficients2
    
    def set_bias(self, bias:float):
        self.bias1 = bias
        
    def set_coefficient1(self, coefficient: list):
        self.coefficient1 = coefficient
            
    def set_coefficient2(self, coefficient: list):
        self.coefficient2 = coefficient
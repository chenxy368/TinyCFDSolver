# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:02:30 2022

@author: HP
"""

def dirichlet_prototype(obj, domain, bias):
    obj[domain] = bias
    return obj

def neumann_prototype(obj, domain, bias, offset, coefficient):
    obj[domain] = coefficient * obj[domain[0] + int(offset[0]), domain[1] + int(offset[1])] + bias
    return obj
    
class BoundaryCondition():
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: list, 
                 boundary_type: str, boundary_opt):
        assert hasattr(boundary_opt, '__call__')
        
        self.boundary_id = boundary_id
        self.boundary_name = boundary_name
        self.boundary_domain = boundary_domain
        
        self.boundary_type = boundary_type
        self.boundary_opt = boundary_opt
    
    def process(self, obj):
        return self.boundary_opt(obj, self.boundary_domain)
    
    def set_domain(self, domain: list):
        self.boundary_domain = domain
    
    def set_opt(self, opt):
        assert hasattr(opt, '__call__')
        self.boundary_opt = opt
    
    def get_id(self):
        return self.boundary_id
    
    def get_name(self):
        return self.boundary_name
    
    def get_domain(self):
        return self.boundary_domain
    
    def get_opt(self):
        return self.boundary_opt
    
    def __str__(self):
        return "ID: " + str(self.boundary_id) + ", Name: " +  self.boundary_name \
                + ", Type: " + self.boundary_type
    
class DirichletBoundary(BoundaryCondition):
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: list, 
                 boundary_type: str, boundary_opt, bias: float):
        assert boundary_type == "Dirichlet" and boundary_opt == dirichlet_prototype
        
        super(DirichletBoundary, self).__init__(boundary_id, boundary_name, boundary_domain, boundary_type, boundary_opt)
        self.bias = bias
        
    def process(self, obj):
        return self.boundary_opt(obj, self.boundary_domain, self.bias)
        
class NeumannBoundary(BoundaryCondition):
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: list, 
                 boundary_type: str, boundary_opt, bias: float, offset: tuple, coefficient: float):
        assert boundary_type == "Neumann" and boundary_opt == neumann_prototype
        
        super(NeumannBoundary, self).__init__(boundary_id, boundary_name, boundary_domain, boundary_type, boundary_opt)
        self.bias = bias
        self.offset = offset
        self.coefficient = coefficient
        
    def process(self, obj):
        return self.boundary_opt(obj, self.boundary_domain, self.bias, self.offset, self.coefficient)
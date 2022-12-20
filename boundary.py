# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:02:30 2022

@author: HP
"""

class boundary_condition():
    def __init__(self, boundary_id: int, boundary_name: str,  boundary_domain: list, boundary_opt):
        assert hasattr(boundary_opt, '__call__')
        
        self.boundary_id = boundary_id
        self.boundary_name = boundary_name
        self.boundary_domain = boundary_domain
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
        return str(self.boundary_id) + " " +  self.boundary_name
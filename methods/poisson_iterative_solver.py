# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:48:39 2022

@author: HP
"""
import numpy as np

class PoissonIterativeSolver():
    def __init__(self, method_info, mesh_data, metrics = None):
        assert metrics.__name__ == "<lambda>"
        
        self.shape = mesh_data[0]
        self.dx = self.read_input("dx", mesh_data[1])
        self.dy = self.read_input("dy", mesh_data[1])
        
        self.interior = self.read_input("domain", mesh_data[2])
        self.exterior = self.read_input("domain_exterior", mesh_data[2])

        self.boundaries = self.read_input("boundary", mesh_data[3])

        self.method_name = method_info[0]
        self.tol = float(method_info[1])  
        
        self.metrics = metrics

    def read_input(self, keyword: str, input_dict: dict):
        if keyword in input_dict:
            return input_dict[keyword]
        else:
            raise RuntimeError(keyword + " MISSING INFORMATION")    
        
class PointJacobi(PoissonIterativeSolver):
    def __init__(self, method_info, mesh_data, metrics = None):
        super(PointJacobi, self).__init__(method_info, mesh_data, metrics)
        
    def solver(self, f):
        psi = np.zeros([self.shape[0], self.shape[1]], dtype = float)

        iteration = 0
        error = 1 

        while error > self.tol:
            for boundary in self.boundaries:
                psi = boundary.process(psi)
                    
            tmp = np.zeros_like(psi)
            tmp[self.interior] = (1.0 / (2.0 / self.dx / self.dx + 2.0 / self.dy / self.dy)) * ((1.0 / self.dx / self.dx) * psi[self.interior[0] - 1, self.interior[1]] \
                                + (1.0 / self.dx / self.dx) * psi[self.interior[0] + 1, self.interior[1]] + (1.0 / self.dy / self.dy) * psi[self.interior[0], self.interior[1] - 1] \
                                    + (1.0 / self.dy / self.dy) * psi[self.interior[0], self.interior[1] + 1] - f[self.interior])
                
            error = self.metrics(tmp, psi, self.interior)
                
            psi[self.interior] = tmp[self.interior]
                
            iteration += 1
        
        print('Number of iterations: ', iteration)
        
        return psi
        
class GaussSeidel(PoissonIterativeSolver):
    def __init__(self, method_info, mesh_data, metrics):
        super(GaussSeidel, self).__init__(method_info, mesh_data, metrics)    

    def solver(self, f):
        psi = np.zeros([self.shape[0], self.shape[1]], dtype = float)

        iteration = 0
        error = 1 
        while error > self.tol:
            for boundary in self.boundaries:
                psi = boundary.process(psi)
                
            tmp = np.zeros_like(psi) 
            tmp[self.exterior] = psi[self.exterior]
            for index in range(len(self.interior[0])):
                node = (self.interior[0][index], self.interior[1][index])
                tmp[node] =  1.0 / (2.0 / self.dx / self.dx + 2.0 / self.dy / self.dy) * ((1.0 / self.dx / self.dx) * tmp[node[0] - 1, node[1]] \
                            + (1.0 / self.dx / self.dx) * psi[node[0] + 1, node[1]] + (1.0 / self.dy / self.dy) * tmp[node[0], node[1] - 1] \
                            + (1.0 / self.dy / self.dy) * psi[node[0], node[1] + 1] - f[node])
            
            
            error = self.metrics(tmp, psi, self.interior)

            psi[self.interior] = tmp[self.interior]
                
            iteration += 1
        
        print('Number of iterations: ', iteration)
        
        return psi

class SOR(PoissonIterativeSolver):
    def __init__(self, method_info, mesh_data, metrics):
        assert len(method_info) == 3 and float(method_info[2]) < 2.0 and float(method_info[2]) > 1.0
        super(SOR, self).__init__(method_info, mesh_data, metrics = None)
        self.wsor = float(method_info[2])  

    def solver(self, f):
        psi = np.zeros([self.shape[0], self.shape[1]], dtype = float)

        iteration = 0
        error = 1 
        
        while error > self.tol:
            for boundary in self.boundaries:
                psi = boundary.process(psi)
                
            tmp = np.zeros_like(psi) 
            tmp[self.exterior] = psi[self.exterior]
            for index in range(len(self.interior[0])):
                node = (self.interior[0][index], self.interior[1][index])
                tmp[node] =  (1.0 - self.wsor) * psi[node] + self.wsor / (2.0 / self.dx / self.dx + 2.0 / self.dy / self.dy) * ((1.0 / self.dx / self.dx) * tmp[node[0] - 1, node[1]] \
                            + (1.0 / self.dx / self.dx) * psi[node[0] + 1, node[1]] + (1.0 / self.dy / self.dy) * tmp[node[0], node[1] - 1] \
                            + (1.0 / self.dy / self.dy) * psi[node[0], node[1] + 1] - f[node])
            
            error = self.metrics(tmp, psi, self.interior)

            psi[self.interior] = tmp[self.interior]
                
            iteration += 1
        
        print('Number of iterations: ', iteration)
        
        return psi

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:48:39 2022

@author: HP
"""
import numpy as np

class PoissonIterativeSolver():
    def __init__(self, shape: tuple, params: list, domains: list, boundary_process, tol, metrics = None):
        assert callable(boundary_process) and callable(metrics) and metrics.__name__ == "<lambda>"
        
        self.shape = shape
        self.dx = params[0]
        self.dy = params[1]
        
        self.interior = domains[0]
        self.exterior = domains[1]

        self.boundary_process = boundary_process

        self.tol = tol
        
        self.metrics = metrics
        
class PointJacobiSolver(PoissonIterativeSolver):
    def __init__(self, shape: tuple, params: list, domains: list, boundary_process, tol, metrics = None):
        super(PointJacobiSolver, self).__init__(shape, params, domains, boundary_process, tol, metrics)
        
    def solver(self, f):
        psi = np.zeros([self.shape[0], self.shape[1]], dtype = float)

        iteration = 0
        error = 1 

        while error > self.tol:
            psi = self.boundary_process(psi)
                    
            tmp = np.zeros_like(psi)
            tmp[self.interior] = (1.0 / (2.0 / self.dx / self.dx + 2.0 / self.dy / self.dy)) * ((1.0 / self.dx / self.dx) * psi[self.interior[0] - 1, self.interior[1]] \
                                + (1.0 / self.dx / self.dx) * psi[self.interior[0] + 1, self.interior[1]] + (1.0 / self.dy / self.dy) * psi[self.interior[0], self.interior[1] - 1] \
                                    + (1.0 / self.dy / self.dy) * psi[self.interior[0], self.interior[1] + 1] - f[self.interior])
                
            error = self.metrics(tmp, psi, self.interior)
                
            psi[self.interior] = tmp[self.interior]
                
            iteration += 1
        
        print('Number of iterations: ', iteration)
        
        return psi
        
class GaussSeidelSolver(PoissonIterativeSolver):
    def __init__(self, shape: tuple, params: list, domains: list, boundary_process, tol, metrics = None):
        super(GaussSeidelSolver, self).__init__(self, shape, params, domains, boundary_process, tol, metrics)

    def solver(self, f):
        psi = np.zeros([self.shape[0], self.shape[1]], dtype = float)

        iteration = 0
        error = 1 
        while error > self.tol:
            psi = self.boundary_process(psi)
                
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

class SORSolver(PoissonIterativeSolver):
    def __init__(self, shape: tuple, params: list, domains: list, boundary_process, tol, metrics = None, wsor = 1.8):
        assert wsor < 2.0 and wsor > 1.0
        super(SORSolver, self).__init__(shape, params, domains, boundary_process, tol, metrics)
        self.wsor = wsor

    def solver(self, f):
        psi = np.zeros([self.shape[0], self.shape[1]], dtype = float)

        iteration = 0
        error = 1 
        
        while error > self.tol:
            psi = self.boundary_process(psi)
                
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

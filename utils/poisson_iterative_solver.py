# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:48:39 2022

@author: HP
"""
import numpy as np

class PoissonIterativeSolver():
    def __init__(self, shape: tuple, interior: list, exterior: list, dx: float, dy: float, metrics = None,
                 tol = 0.1, boundaries = []):
        assert len(shape) == 2  and callable(metrics) and metrics.__name__ == "<lambda>"
        
        self.shape = shape
        self.interior = interior
        self.exterior = exterior
        self.dx = dx
        self.dy = dy

        self.tol = tol
        self.metrics = metrics
        self.boundaries = boundaries

        
class PointJacobi(PoissonIterativeSolver):
    def __init__(self, shape: tuple, interior: list, exterior: list, dx: float, dy: float, metrics = None,
                 tol = 0.1, boundaries = []):
        super(PointJacobi, self).__init__(shape, interior, exterior, dx, dy, metrics, tol, boundaries)
        
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
    def __init__(self, shape: tuple, interior: list, exterior: list, dx: float, dy: float, metrics = None,
                 tol = 0.1, boundaries = []):
        super(GaussSeidel, self).__init__(shape, interior, exterior, dx, dy, metrics, tol, boundaries)    

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
    def __init__(self, shape: tuple, interior: list, exterior: list, dx: float, dy: float, metrics = None,
                 tol = 0.1, boundaries = [],  wsor = 1.8):
        super(SOR, self).__init__(shape, interior, exterior, dx, dy, metrics, tol, boundaries)
        self.wsor = wsor

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

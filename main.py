# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 03:12:39 2022

@author: Xinyang Chen
"""
import numpy as np
import sys
import os
import argparse

from methods import FracStep, StreamFunctionVorticity, PoissonIterative, UpwindCentral2D, CrankNicolson2D, ADI

method_dict = {
        "FracStep": FracStep,
        "SFV": StreamFunctionVorticity, 
        "PoissonIterative": PoissonIterative,
        "UpwindCentral2D": UpwindCentral2D, 
        "CrankNicolson2D": CrankNicolson2D,
        "ADI": ADI
    }

def main(path, params, save=False):
    method_name = "FracStep"
    line_num = 0
    with open(path + "/grid_information.txt") as file:
        for line in file:
            if line_num == 0 and line.split()[0] != "METHOD":
                raise RuntimeError("Incorrect format")
                
            if line_num == 0:
                line_num += 1
                continue
            
            str_list = line.split()
            if str_list[0] not in method_dict:
                raise RuntimeError("Unknown Method")
            method_name = str_list[0]
            break
        
    init_path = path + "/init.npy"
    init = None
    if os.path.exists(init_path):
        init = np.load(init_path)
    
    sys.path.append(path)
    from lambda_functions import lambda_list
    method = method_dict[method_name](path, lambda_list, init)        
    sys.path.remove(path)
    res = method.solve(params)
    
    if save:
        np.save(path + "/res.npy", res)
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = str, default = '', help = 'project path')
    parser.add_argument('--params', nargs='*', type=float, default=0.0, help='parameters for solve')
    parser.add_argument('--save', action='store_true', help='save result or not')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
    #main("sample_cases/ADI/diffusion_ADI_obstacle_case", (int(0.3/0.001), 5), False)
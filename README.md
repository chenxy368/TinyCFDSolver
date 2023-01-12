# TinyCFDSolver
A tiny python computational fluid dynamics solver with several finite difference methods by Xinyang CHEN(Thanks Wenzhuo XU's help in implementing fractional step method).  
<p align="center">
  <img width="460" height="300" src="sample_cases/frac_step/logo_case/result/pressure.gif">
</p>


## Overview
The TinyCFDSolver's solvers are implemented in solvers folder. The methods (mediator classes) are implemented in methods folder. Boundary classes, plot methods and grid loaders are in utils folder. Several example cases can be found in the sample cases folder. To test the cases, the command line should follow:
```shell
# For poisson_iterative, the parameter is the right hand side of the poisson equation, which can be a float or a path.
python main.py --path sample_cases/poisson_iterative/heat_diffusion_case --params 0
# For other methods, the first parameter is the number of timesteps, and the second parameter is the check point interval. At each checkpoint, methods cache one result and also call step process funtions. 
python main.py --path sample_cases/ADI/diffusion_ADI_obstacle_case --params 30 5
# If you want to save final results as a .npy file
python main.py --path sample_cases/ADI/diffusion_ADI_obstacle_case --params 30 5 --save
```
## Requirements
```shell
pip install -r requirements.txt
```

## Demos
Fractional step method solving N-S equation:
<p align="center">
  <img width="345" height="225" src="sample_cases/frac_step/box_obstacle_case/result/speed.gif">
  <img width="345" height="225" src="sample_cases/frac_step/box_obstacle_case/result/pressure.gif">
</p>

<p align="center">
  <img width="345" height="225" src="sample_cases/frac_step/half_case/result/speed.gif">
  <img width="345" height="225" src="sample_cases/frac_step/half_case/result/pressure.gif">
</p>

Stream function vorticity method solving N-S equation:
<p align="center">
  <img width="345" height="225" src="sample_cases/SFV/driven_cavity_case/result/speed.gif">
  <img width="345" height="225" src="sample_cases/SFV/flow_obstacle_case/result/speed.gif">
</p>

Poisson iterative solver solving heat diffusion:
<p align="center">
  <img width="345" height="225" src="sample_cases/poisson_iterative/heat_diffusion_case/result/temperature.png">
</p>

Upwind central scheme solving advection diffusion equation:
<p align="center">
  <img width="345" height="225" src="sample_cases/upwind_central/advection_diffusion_case/result/temperature.gif">
  <img width="345" height="225" src="sample_cases/upwind_central/advection_diffusion_init_case/result/temperature.gif">
</p>

Crank Nicolson implicit method solving diffusion equation:
<p align="center">
  <img width="345" height="225" src="sample_cases/crank_nicolson/diffusion_CK_case/result/temperature.gif">
  <img width="345" height="225" src="sample_cases/crank_nicolson/diffusion_CK_obstacle_case/result/temperature.gif">
</p>

ADI method solving diffusion equation:
<p align="center">
  <img width="345" height="225" src="sample_cases/ADI/diffusion_ADI_case/result/temperature.gif">
  <img width="345" height="225" src="sample_cases/ADI/diffusion_ADI_obstacle_case/result/temperature.gif">
</p>

## Project Files
Each project requires four setting files(e.g. sample_cases/upwind_central/advection_diffusion_case):  
1. gird_information.txt(Do not support comment currently)
```shell
METHOD #name of the method and parameters for it
UpwindCentral2D
PARAMETER #parameters
dx 0.01
dy 0.01
dt 0.002
u 1.5
v 1
molecular_diffusivity_x 0.001
molecular_diffusivity_y 0.001
source_temparature 1
DOMAIN #domain information, in this case only interior grid ID
mesh -1
BOUNDARY #boundaries, group gridID name type parameterlist
X 0 Wall_Neumman Linear 1 0 -1 0
X 1 Wall_Dirichlet Const 0
X 2 Heat_Source Const source_temparature
X 3 Wall_Neumman Linear 1 0 1 0
X 4 Wall_Neumman Linear 1 -1 0 0
```
2. mesh.csv
```shell
# A csv file store grid ID, the solver and boundary objects can get their position by getting slice with grid ID.
0,0,0,0,0,0, ... ,0
1,-1,-1,-1,-1, ... ,4
1,-1,-1,-1,-1, ... ,4
...
1,-1,-1,-1,-1, ... ,4
3,3,3,3,3,3,3, ... ,3
```
3. lambda_functions.py
```shell
# Define all required lambda function of methods in this file. (e.g. visualization and metrics functions)
from utils import plot_one_contourf, animate
# Debug ploter
ploter = lambda X, dx, dy, dt, t: (plot_one_contourf(X.transpose(), dx, dy, "temperature at " + str(round((t + 1) * dt, 3)) 
+ "s", "temperature[$^\circ$C]", 0.0, 1.05))        
animator = lambda X, dx, dy: (animate(X, dx, dy, "temperature", "temperature[$^\circ$C]", 0.0, 1.0))
lambda_list = [ploter, animator]
```

4. init.npy(optional), initial conditions

## Structure
These slides briefly show how this project is designed and implemented.
<p align="center">
  <img width="800" height="400" src="readme_image/slide1.png">
</p>
<p align="center">
  <img width="800" height="450" src="readme_image/slide2.png">
</p>
<p align="center">
  <img width="800" height="450" src="readme_image/slide3.png">
</p>
<p align="center">
  <img width="800" height="450" src="readme_image/slide4.png">
</p>
<p align="center">
  <img width="800" height="450" src="readme_image/slide5.png">
</p>

## Implement New Boundary
E.g. const boundary class
1. Define the form in the grid_information.txt, and how this boundary class modify the target array.
```shell
# Group GridID Name Type Params
X 1 Wall_Dirichlet Const 0
arr[boundary_postion] = const
```
2. Implement a new boudary class(inherit a base boundary class)
```shell
class ConstCondition(BoundaryCondition):
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
```
3. Add to the boundary dictionary in gridloader.py
```shell
boundary_classes = {"Const": ConstCondition,
                    "Linear": LinearCondition, 
                    "LinearCombination": LinearCombinationCondition, 
                    "NQuantityLinearCombination": NQuantityLinearCombinationCondition,
                    "LinearSpacial": LinearSpacialCondition}
```
4. In method class the boundary object are grouped by its group name(e.g. "X", X 1 Wall_Dirichlet Const 0)
```shell
class FracStep():
    def __init__(self, root, lambda_list: list, initial_condition = None):
        """ Inits FracStep class with root of the project, lambda functions list and initial conditions"""
        ...
        method_info, mesh_data, _ = loader.load_grid(domain_dict, mesh_boundary_dict)
        
        # Get boundaries
        self.u_boundaries = mesh_data[3]["u"]
        self.v_boundaries = mesh_data[3]["v"]
        self.p_boundaries = mesh_data[3]["p"]
        ...
        # Initialize Solver
        self.solver = FracStepSolver((mesh_data[0]["u"], mesh_data[0]["v"], mesh_data[0]["p"]), 
                                     (mesh_data[1]["dt"], mesh_data[1]["dx"], mesh_data[1]["dy"], 
                                      mesh_data[1]["kinematic_viscosity"], mesh_data[1]["density"]), 
                                     (mesh_data[2]["u"], mesh_data[2]["v"], mesh_data[2]["p"], 
                                      mesh_data[2]["u_exterior"], mesh_data[2]["v_exterior"], mesh_data[2]["p_exterior"]),
                                     (self.u_boundary_process, self.v_boundary_process, self.p_boundary_process_frac_step),
                                     poisson_solver, self.extra_computing, step_visualization, final_visualization, initial_condition)

    def solve(self, params: list): 
        ...
    # Solvers will call these boundary process functions and the boundary objects in different groups will be processes separately.
    def u_boundary_process(self, u, v, p, t):
        for boundary in self.u_boundaries:
            u = boundary.process(u)
        return u
    
    def v_boundary_process(self, u, v, p, t):
        for boundary in self.v_boundaries:
            v = boundary.process(v)
        return v
    
    def p_boundary_process_possion_iterative(self, p):
        for boundary in self.p_boundaries:
            p = boundary.process(p)
        return p
    
    def p_boundary_process_frac_step(self, u, v, p, t):
        pass
```

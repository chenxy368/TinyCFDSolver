# TinyCFDSolver
A tiny python computational fluid dynamics solver with several finite difference methods by Xinyang CHEN(Thanks Wenzhuo XU's help implement fractional step method).  
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

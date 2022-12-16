# TinyCFDSolver
A multi-domain solver with Fractional Step method discretization in space and first order explicit in time by Xinyang CHEN and Wenzhuo XU.
## Overview
The Fractional Step solver is implemented in `NSfracStep_single_phase.py`, while a generalized Poisson solver is implemented in `solvers.py`. Several example grids can also be found in the folder. To implement your own grid, reference the example grid files for exact locations of ghost nodes and boundaries. 
## Use the code
```shell
pip install -r requirements.txt
python NSfracStep_single_phase.py
```

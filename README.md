# Rate–Distortion Optimization
This repo contains code to experiment with our convex formulation of rate–distortion (R–D) discrete sources.

## Files
- `main_algo.py`  
  Core functions to:
  - Build distortion matrices  
  - Solve the rate–distortion problem for given target distortions constraints (global and subsets)
  - Run the iterative codebook update algorithm  
  - Plot R–D curves and basic diagnostics  

- `executions.ipynb`  
  Example notebook that:
  1. Defines source distributions and reconstruction points (codebooks)  
  2. Runs the R–D solver over a range of target distortions  
  3. Run the R-D solver with mixed fidelity constraints and demonstrate dual variables analysis
  4. Runs the main iterative algorithm to refine the codebook  
  5. Plots the resulting R–D curves and convergence behavior  
  6. Compare our solutions with Lloyd-max

## How to Use
Open and run `executions.ipynb` to reproduce the experiments and plots (up to random initialization)
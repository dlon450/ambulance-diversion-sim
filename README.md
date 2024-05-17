# Data-Adaptive Ambulance Diversion via ED Simulation
We focus on ambulance diversions in the emergency department (ED) as a prototype example to illustrate how data-adaptive decision-making, via the proposed digital twinning approach, can lead to better mortality outcomes. Diverting ambulances aims to reduce ED overcrowding and balance patient load among multiple EDs. However, setting a static policy, such as diverting only when the number of patients in the ED (queue) exceeds a prespecified threshold, can be highly suboptimal during surge situations, because the threshold used in normal situations can perform poorly in the latter cases. If we can adaptively change the threshold level by assimilating and optimizing according to the predicted surge, we could more effectively avoid overcrowding and potentially save lives. This simple prototype example showcases the mortality improvement using such a data-adaptive threshold policy. In particular, the optimization of this data-adaptive policy requires running system simulation for the mortality outcomes from each hospital. This simulation model resembles the proposed digital twins that are envisioned to build at a much larger scale. Moreover, in this example, a simulation run of each trajectory to output the mortality outcome already takes ~4 hours on a computer with an Intel Core i5-8250U CPU and 8GB RAM. Optimizing an adaptive policy at each threshold-changing opportunity would require running a sufficient number (e.g., 100) of trajectories to wash out the simulation error, as well as evaluating a wider grid of threshold values for performance comparison, thus altogether leading to over 500 hours. This motivates us to investigate a metamodeling of the digital twin via a computationally much lighter AI model that can be optimized more straightforwardly, thus leading to an overall AI-digital-twinning integrative approach.
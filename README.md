## AFI-PINNs
This is the code for the paper "Failure-informed adaptive sampling for PINNs, part II: combining with re-sampling and subset simulation".  
## Abstract
Physic-informed neural networks are a powerful tool to solve partial differential equations (PDEs). While it generally suffers great computational cost and is difficult to be generalized to more practical applications. In this paper, we choose to apply the failure informed PINNs with the subset simulation as the sampling strategy to improve the performance of tradition PINN solvers. Furthermore, we develop a new adaptive sampling method to update the collocation dataset in an annealing manner while keeping the size of training dataset invariant, which can greatly save computational cost. We see that with different performance functions, the novel method can be applied to different scenarios, which is hard to solve using common adaptive sampling strategies. We further implement several numerical experiments to verify the effectiveness of our method. 


## Citation

```
  @article{https://doi.org/10.1007/s42967-023-00312-7,
        author = {Gao, Zhiwei and Tang, Tao and Yan, Liang and Zhou, Tao},
        title = {Failure-Informed Adaptive Sampling for PINNs, Part II: Combining with Re-sampling and Subset Simulation},
        journal = {Commun. Appl. Math. Comput.},
        volume = {6},
        pages = {1720-1741},
        year = {2024}}
```

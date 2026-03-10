## References 

### General

[1] - Ragan-Kelley, J., Barnes, C., Adams, A., Paris, S., Durand, F., & Amarasinghe, S. (2013). Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. Acm Sigplan Notices, 48(6), 519-530. <br/>
+ See video below

[2] - Li, M., Liu, Y., Liu, X., Sun, Q., You, X., Yang, H., ... & Qian, D. (2020). The deep learning compiler: A comprehensive survey. IEEE Transactions on Parallel and Distributed Systems, 32(3), 708-727.

### TVM
[3] - Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Shen, H., ... & Krishnamurthy, A. (2018). {TVM}: An automated {End-to-End} optimizing compiler for deep learning. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18) (pp. 578-594).

[4] - Feng, S., Hou, B., Jin, H., Lin, W., Shao, J., Lai, R., ... & Chen, T. (2023, January). Tensorir: An abstraction for automatic tensorized program optimization. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (pp. 804-817).

[5] - Lai, R., Shao, J., Feng, S., Lyubomirsky, S., Hou, B., Lin, W., ... & Chen, T. (2025, March). Relax: Composable abstractions for end-to-end dynamic machine learning. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (pp. 998-1013).

### Auto-tuning

[6] - Chen, T., Zheng, L., Yan, E., Jiang, Z., Moreau, T., Ceze, L., ... & Krishnamurthy, A. (2018). Learning to optimize tensor programs. Advances in Neural Information Processing Systems, 31.

[7] - Zheng, L., Jia, C., Sun, M., Wu, Z., Yu, C. H., Haj-Ali, A., ... & Stoica, I. (2020). Ansor: Generating {High-Performance} tensor programs for deep learning. In 14th USENIX symposium on operating systems design and implementation (OSDI 20) (pp. 863-879).

[8] - Shao, J., Zhou, X., Feng, S., Hou, B., Lai, R., Jin, H., ... & Chen, T. (2022). Tensor program optimization with probabilistic programs. Advances in Neural Information Processing Systems, 35, 35783-35796.

### Reinforcement Learning for auto-tuning

[9] - Zhang, Z., He, B., & Zhang, Z. (2022, August). Harl: Hierarchical adaptive reinforcement learning based auto scheduler for neural networks. In Proceedings of the 51st International Conference on Parallel Processing (pp. 1-13).
 
## Other learning material
- An interesting description of optimization and Halide : https://www.youtube.com/watch?v=1ir_nEfKQ7A&t=126s
- A visualization of cache and tiling effects on matrix multiplication : https://arthurviens.github.io/cache-visualizer/

## Tools
### Apache TVM : https://tvm.apache.org/ <br/>
Installation https://tvm.apache.org/docs/install/index.html
- Installation on Linux is very highly recommended (possibly Docker)
- If on windows, installation on Docker is highly recommended. Else, good luck !
- For profiling, [building TVM with PAPI support](https://tvm.apache.org/docs/v0.8.0/how_to/profile/papi.html) is recommended.

Extremely good tutorial : [MLC Course](https://book.mlc.ai/) (chapters 1 to 4)


## Internship approximative planning
Approximative planning to have vision over the next weeks, **however**, you're free to deviate from it.

* Weeks 1 & 2 : Learning the context, reading general bibliography, install the environment and the tools.
* Weeks 3 & 4 : Experiments regarding code transformations, and TVM's auto-tuning + reading auto-tuning related bibliography. 
* Weeks 5 & 6 : Experiments regarding auto-tuning with a reinforcement algorithm + reading reinforcement learning for auto-tuning related bibliography
* Weeks 7 : Analysis of the experiments results, what works, what doesn't and why, report redaction.
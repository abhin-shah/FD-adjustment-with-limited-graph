# Source code for "Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge"

Reference: Abhin Shah, Karthikeyan Shanmugam, Murat Kocaoglu,
"Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge," 
37th Conference on Neural Information Processing Systems (NeurIPS), 2023

Contact: abhin@mit.edu

Arxiv: https://arxiv.org/pdf/2306.11008

### Dependencies:

In order to successfully execute the code, the following libraries must be installed:

1. Python --- numpy, pandas, networkx, scikit-learn, scipy, causal-learn, rpy2, matplotlib, pickle, argparse, time, itertools

2. R --- RCIT, reticulate, RcppCNPy, PAGId, pcalg, dagitty

### Command inputs:

-   niter: number of iterations (default = 100)
-   nobs: list of number of observed variables
-   d: expected in-degree of variables (default = 2)
-   qq: probability parameter for confounders (default = 0.1)
-   table: table number (1 or 2) - determines treatment selection criteria (default = 1)
-   K: maximum subset size for bounded search (default = 5)
-   nr: number of repetitions for averaging (default = 10)
-   all_samples: total number of samples for data generation (default = 100000)
-   samples_list: list of sample sizes for ATE estimation

### Reproducing the figures and tables:

1. To reproduce Table 1, run the following commands:
```shell
$ mkdir table1_iter100_q0.0
$ python3 -W ignore generate_graphs.py --nobs 10 15 --d 2 --qq 0.0 --table 1
$ python3 -W ignore fd_hit_rate.py --nobs 10 15 --d 2 --qq 0.0 --table 1
$ python3 -W ignore fd_subset_hit_rate.py --nobs 10 15 --d 2 --K 5 --qq 0.0 --table 1
$ mkdir table1_iter100_q0.5
$ python3 -W ignore generate_graphs.py --nobs 10 15 --d 3 --qq 0.5 --table 1
$ python3 -W ignore fd_hit_rate.py --nobs 10 15 --d 3 --qq 0.5 --table 1
$ python3 -W ignore fd_subset_hit_rate.py --nobs 10 15 --d 3 --K 5 --qq 0.5 --table 1
$ mkdir table1_iter100_q1.0
$ python3 -W ignore generate_graphs.py --nobs 10 15 --d 4 --qq 1.0 --table 1
$ python3 -W ignore fd_hit_rate.py --nobs 10 15 --d 4 --qq 1.0 --table 1
$ python3 -W ignore fd_subset_hit_rate.py --nobs 10 15 --d 4 --K 5 --qq 1.0 --table 1
```
To test baseline methods for comparison, edit table parameter in baseline_hit_rate.R and run:
```shell
$ Rscript baseline_hit_rate.R
```

2. To reproduce Table 2, run the following commands:
```shell
$ mkdir table2_iter100_q0.0
$ python3 -W ignore generate_graphs.py --nobs 10 15 --d 2 --qq 0.0 --table 2
$ python3 -W ignore fd_hit_rate.py --nobs 10 15 --d 2 --qq 0.0 --table 2
$ python3 -W ignore fd_subset_hit_rate.py --nobs 10 15 --d 2 --K 5 --qq 0.0 --table 2
$ mkdir table2_iter100_q0.5
$ python3 -W ignore generate_graphs.py --nobs 10 15 --d 3 --qq 0.5 --table 2
$ python3 -W ignore fd_hit_rate.py --nobs 10 15 --d 3 --qq 0.5 --table 2
$ python3 -W ignore fd_subset_hit_rate.py --nobs 10 15 --d 3 --K 5 --qq 0.5 --table 2
$ mkdir table2_iter100_q1.0
$ python3 -W ignore generate_graphs.py --nobs 10 15 --d 4 --qq 1.0 --table 2
$ python3 -W ignore fd_hit_rate.py --nobs 10 15 --d 4 --qq 1.0 --table 2
$ python3 -W ignore fd_subset_hit_rate.py --nobs 10 15 --d 4 --K 5 --qq 1.0 --table 2
```
To test baseline methods for comparison, edit table parameter in baseline_hit_rate.R and run:
```shell
$ Rscript baseline_hit_rate.R
```

3. To reproduce Figure 5, run the following commands:
```shell
$ python3 -W ignore fd_causal_effect.py --nobs 10 --samples_list 100 1000 10000 --d 2 --qq 1.0 --table 1
```

4. To reproduce Figure 6, run the following command:
```shell
$ python -W ignore German.py
```

5. To reproduce Figure 9, run the following commands:
```shell
$ mkdir synthetic_1
$ python3 -W ignore synthetic.py --graph 1 --nr 50 -d "4"
$ mkdir synthetic_2
$ python3 -W ignore synthetic.py --graph 2 --nr 50 -d "4"
$ mkdir synthetic_3
$ python3 -W ignore synthetic.py --graph 3 --nr 50 -d "4"
```

6. To reproduce Section 3.1 demonstration, run the following command:
```shell
$ python3 -W ignore different_formula.py
```

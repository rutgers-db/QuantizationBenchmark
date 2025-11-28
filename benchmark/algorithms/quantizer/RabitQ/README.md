# RabitQ Source Code
During the search phase (fast\_scan function), `tmp_dist` estimates the d(o, q), `error_bound` is an error bound presented in Formula 16 in the paper.
We separate ordinary search and re-ranking process
In the ordinary search process, we use the estimated distance to search.
In the re-rank process, RabitQ design a pruning strategy: minus estimated distance by the error bound to get a lower-bound. If the lower-bound is larger than the current top-k distance, it will not calculate the exact distance
To calculate the quantization error (MSE), we use the database vectors as search queries and found their corresponding estimated distance through search.

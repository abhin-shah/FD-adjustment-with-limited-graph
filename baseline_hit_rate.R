# Reference: A Shah, K Shanmugam, M Kocaoglu
# "Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge,"
# In 37th Conference on Neural Information Processing Systems (NeurIPS), 2023
#
# Last updated: December 15, 2023
# Code author: Abhin Shah
#
# File name: baseline_hit_rate.R
#
# Description: Compute hit rates for baseline PAG-based methods (Tables 1-2).
# Tests IDP algorithm on PAGs to identify causal effects for comparison
# with proposed front-door adjustment method.

source("supporting_func.R")
require("reticulate")
library(RcppCNPy)
library(PAGId)
source_python("pickle_reader.py")

number_observed_list <- c(15)
number_iterations <- 100
degree <- 4
qq <- 0.0
table <- 1  # 1 for Table 1 (strict), 2 for Table 2 (relaxed)

output_path <- paste0('table', table, '_iter', number_iterations, '_q', sprintf("%.1f", qq), '/[', paste(number_observed_list, collapse = ', '), ']_d_', degree)

baseline_CL <- matrix(0, nrow = length(number_observed_list), ncol = number_iterations)

all_t <- npyLoad(paste0(output_path, '_t.npy'))
all_y <- npyLoad(paste0(output_path, '_y.npy'))
all_num_con <- npyLoad(paste0(output_path, '_num_con.npy'))
all_A <- read_pickle_file(paste0(output_path, '_A.bin'))
all_A_pag <- read_pickle_file(paste0(output_path, '_A_pag.bin'))

starting_time <- Sys.time()

for (n_count in seq_along(number_observed_list)) {
  if (n_count >= 1) {
    cat(paste("n_count: ", n_count, "\n"))
    number_obs = number_observed_list[n_count]
    if (number_obs <= 15){
      for (iter in seq_len(number_iterations)) {
        if (iter >= 1) {
          if ((iter) %% 10 == 0) {
            cat(paste("iter: ", iter, "\n"))
          }
          t <- all_t[n_count, iter] + 1
          y <- all_y[n_count, iter] + 1
          number_con <- all_num_con[n_count, iter]
          p <- number_obs + number_con
          A_total <- all_A[[n_count]][[iter]]
          
          adag <- getDAG(t, y, number_obs, number_con, A_total)
          A_pag_R <- all_A_pag[[n_count]][[iter]]
          colnames(A_pag_R) <- rownames(A_pag_R) <- as.character(1:number_obs)
          retPAG_CL <- IDP(A_pag_R, adag$x, adag$y, verbose = FALSE)
          baseline_CL[n_count, iter] <- as.numeric(retPAG_CL$id)
        }
      }
    }
  }
}

saveRDS(Sys.time() - starting_time, paste0(output_path, '_time_base_CL.RDS'))
cat(paste('Run time for base CL method: ', Sys.time() - starting_time, '\n'))
write.table(baseline_CL, file = paste0(output_path, '_baseline_CL.csv'), row.names = FALSE, col.names = FALSE)
print(rowSums(baseline_CL)/number_iterations)
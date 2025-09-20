# Reference: A Shah, K Shanmugam, M Kocaoglu
# "Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge,"
# In 37th Conference on Neural Information Processing Systems (NeurIPS), 2023
#
# Last updated: December 15, 2023
# Code author: Abhin Shah
#
# File name: supporting_func.R
#
# Description: Supporting R functions for causal graph analysis and PAG construction.
# Contains utilities for converting between graph representations and testing methods.

getDAG <- function(t, y, number_obs, number_con, A_total)  {
  # Convert adjacency matrix to dagitty DAG object with specified treatment and outcome
  p <- number_obs + number_con
  allvars <- allvars <- as.character(1:p)
  colnames(A_total) <- rownames(A_total) <- allvars
  lat <- ""
  x <- as.character(t:t)
  y <- as.character(y:y)
  dagg <- pcalg::pcalg2dagitty(t(A_total), colnames(A_total), type="dag")
  dagitty::exposures(dagg) <- x
  dagitty::outcomes(dagg) <- y
  return(list(A_total=A_total, x=x, y=y, lat=lat, dagg=dagg))
}

getPAG <- function(dag, verbose = FALSE) {
  # Generate PAG from DAG using FCI algorithm with high confidence level
  valR <- FALSE
  while(!valR) {
    R <- dagitty::impliedCovarianceMatrix(dag, b.default = NULL, b.lower = -0.6,
                                          b.upper = 0.6, eps = 1, standardized = TRUE)
    
    R <- round(R, 14)
    valR <- matrixcalc::is.symmetric.matrix(R) &&
      matrixcalc::is.positive.definite(R, tol=1e-8)
    if(verbose)
      cat(paste0("valR=", valR, "\n"))
  }
  
  latR <- R
  suffStat = list(C = latR, n = 10^9)
  
  true.pag <- pcalg::fci(suffStat,
                         indepTest = pcalg::gaussCItest, #p = ncol(-latR),
                         labels= colnames(suffStat$C), alpha = 0.9999)
  return(true.pag)
}

## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse  = TRUE,
  comment   = "#>",
  fig.width = 6,
  fig.height = 4,
  out.width = "95%"
)

## ----data---------------------------------------------------------------------
if (!requireNamespace("MASS", quietly = TRUE))
  stop("Please install the MASS package: install.packages('MASS')")

library(SWORD)
data(Boston, package = "MASS")

set.seed(42)
n     <- nrow(Boston)
train <- sample(n, floor(0.8 * n))
test  <- setdiff(seq_len(n), train)

X_train <- Boston[train, setdiff(names(Boston), "medv")]
y_train <- Boston$medv[train]
X_test  <- Boston[test,  setdiff(names(Boston), "medv")]
y_test  <- Boston$medv[test]

cat("Training set:", nrow(X_train), "obs |",
    "Test set:", nrow(X_test), "obs\n")

## ----tors-fit-----------------------------------------------------------------
set.seed(1)
tree <- TORS(X_train, y_train, nmin = 15, cp = 0.02)
print(tree)

## ----tors-text, fig.height=5--------------------------------------------------
plot_tors_text(tree, top_k = 3)

## ----tors-plot----------------------------------------------------------------
plot(tree)

## ----tors-cp------------------------------------------------------------------
set.seed(1)
t_deep     <- TORS(X_train, y_train, nmin = 10, cp = 0.000)
t_moderate <- TORS(X_train, y_train, nmin = 10, cp = 0.010)
t_shallow  <- TORS(X_train, y_train, nmin = 10, cp = 0.050)

cp_tbl <- data.frame(
  cp        = c(0.000, 0.010, 0.050),
  n_nodes   = c(t_deep$n_nodes,     t_moderate$n_nodes,     t_shallow$n_nodes),
  n_leaves  = c(sum(t_deep$is_leaf), sum(t_moderate$is_leaf), sum(t_shallow$is_leaf)),
  max_depth = c(max(t_deep$depth),   max(t_moderate$depth),   max(t_shallow$depth))
)
print(cp_tbl, row.names = FALSE)

## ----tors-ntopcor-------------------------------------------------------------
set.seed(1)
t_nc2 <- TORS(X_train, y_train, nmin = 15, cp = 0.02, n_top_cor = 2)
t_nc4 <- TORS(X_train, y_train, nmin = 15, cp = 0.02, n_top_cor = 4)

cat("--- n_top_cor = 2 (default, sparse oblique) ---\n")
plot_tors_text(t_nc2, top_k = 2)
cat("\n--- n_top_cor = 4 (richer oblique splits) ---\n")
plot_tors_text(t_nc4, top_k = 4)

## ----sword-fit----------------------------------------------------------------
set.seed(42)
forest <- SWORD(X_train, y_train,
                m       = 50,
                nmin    = 10,
                cp      = 0.01,
                oob     = TRUE,
                verbose = FALSE)
print(forest)

## ----oob-convergence----------------------------------------------------------
plot(forest, which = "oob")

## ----oob-scatter--------------------------------------------------------------
plot_oob_fit(forest, y_train)

## ----predict------------------------------------------------------------------
y_hat <- predict(forest, X_test)

rmse <- sqrt(mean((y_hat - y_test)^2))
mae  <- mean(abs(y_hat - y_test))
r2   <- cor(y_hat, y_test)^2

cat(sprintf("Test RMSE : %.3f\n", rmse))
cat(sprintf("Test MAE  : %.3f\n", mae))
cat(sprintf("Test R^2  : %.3f\n", r2))

## ----predict-full-df----------------------------------------------------------
y_hat2 <- predict(forest, Boston[test, ])  # medv column is silently dropped
all.equal(y_hat, y_hat2)

## ----vi-----------------------------------------------------------------------
vi <- VI_SWORD(forest)
print(round(vi, 2))

## ----vi-plot------------------------------------------------------------------
plot_vi_sword(forest, top_n = 13)

## ----pdp-2var, fig.height=4, fig.width=10, out.width="100%"-------------------
old_par <- par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
pdp_sword(forest, X_train, "lstat", main = "PDP: lstat")
pdp_sword(forest, X_train, "rm",    main = "PDP: rm")
par(old_par)

## ----pdp-grid, fig.height=7, fig.width=10, out.width="100%"-------------------
top4 <- names(vi)[1:4]
old_par <- par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
for (v in top4)
  pdp_sword(forest, X_train, v, main = paste("PDP:", v))
par(old_par)

## ----params-helper, include=FALSE---------------------------------------------
rmse_f <- function(y, yhat) sqrt(mean((y - yhat)^2))
p      <- ncol(X_train)

## ----params-relation----------------------------------------------------------
set.seed(42)
f_pearson <- SWORD(X_train, y_train, m = 20, nmin = 10, cp = 0.01,
                   relation = "Pearson", verbose = FALSE)
set.seed(42)
f_mi      <- SWORD(X_train, y_train, m = 20, nmin = 10, cp = 0.01,
                   relation = "MI",      verbose = FALSE)

cat(sprintf("Pearson  — test RMSE: %.3f\n", rmse_f(y_test, predict(f_pearson, X_test))))
cat(sprintf("MI       — test RMSE: %.3f\n", rmse_f(y_test, predict(f_mi,      X_test))))

## ----params-weight------------------------------------------------------------
set.seed(42)
f_scale  <- SWORD(X_train, y_train, m = 20, nmin = 10, cp = 0.01,
                  weight_scheme = "scale",  verbose = FALSE)
set.seed(42)
f_robust <- SWORD(X_train, y_train, m = 20, nmin = 10, cp = 0.01,
                  weight_scheme = "robust", verbose = FALSE)

cat(sprintf("scale    — test RMSE: %.3f\n", rmse_f(y_test, predict(f_scale,  X_test))))
cat(sprintf("robust   — test RMSE: %.3f\n", rmse_f(y_test, predict(f_robust, X_test))))

## ----params-rfvar-------------------------------------------------------------
set.seed(42)
f_full <- SWORD(X_train, y_train, m = 20, nmin = 10, cp = 0.01,
                rf_var = p,               verbose = FALSE)  # all predictors
set.seed(42)
f_half <- SWORD(X_train, y_train, m = 20, nmin = 10, cp = 0.01,
                rf_var = floor(p / 2),    verbose = FALSE)  # half
set.seed(42)
f_sqrt <- SWORD(X_train, y_train, m = 20, nmin = 10, cp = 0.01,
                rf_var = floor(sqrt(p)),  verbose = FALSE)  # sqrt rule

cat(sprintf("rf_var = %2d (all)        — test RMSE: %.3f\n",
            p,               rmse_f(y_test, predict(f_full, X_test))))
cat(sprintf("rf_var = %2d (floor p/2)  — test RMSE: %.3f\n",
            floor(p / 2),   rmse_f(y_test, predict(f_half, X_test))))
cat(sprintf("rf_var = %2d (floor sqrt) — test RMSE: %.3f\n",
            floor(sqrt(p)), rmse_f(y_test, predict(f_sqrt, X_test))))

## ----params-rand--------------------------------------------------------------
set.seed(42)
f_rnd_t <- SWORD(X_train, y_train, m = 20, nmin = 10, cp = 0.01,
                 rand_ntopcor = TRUE,  verbose = FALSE)
set.seed(42)
f_rnd_f <- SWORD(X_train, y_train, m = 20, nmin = 10, cp = 0.01,
                 rand_ntopcor = FALSE, verbose = FALSE)

cat(sprintf("rand_ntopcor = TRUE  — test RMSE: %.3f\n",
            rmse_f(y_test, predict(f_rnd_t, X_test))))
cat(sprintf("rand_ntopcor = FALSE — test RMSE: %.3f\n",
            rmse_f(y_test, predict(f_rnd_f, X_test))))

## ----params-table, echo=FALSE-------------------------------------------------
tbl <- data.frame(
  Parameter = c(
    "relation", "relation",
    "weight_scheme", "weight_scheme",
    "rf_var", "rf_var", "rf_var",
    "rand_ntopcor", "rand_ntopcor"
  ),
  Value = c(
    "Pearson (default)", "MI",
    "scale (default)", "robust",
    paste0("p = ", p, " — all (default)"),
    paste0("floor(p/2) = ", floor(p / 2)),
    paste0("floor(sqrt(p)) = ", floor(sqrt(p))),
    "TRUE (default)", "FALSE"
  ),
  Test_RMSE = round(c(
    rmse_f(y_test, predict(f_pearson, X_test)),
    rmse_f(y_test, predict(f_mi,      X_test)),
    rmse_f(y_test, predict(f_scale,   X_test)),
    rmse_f(y_test, predict(f_robust,  X_test)),
    rmse_f(y_test, predict(f_full,    X_test)),
    rmse_f(y_test, predict(f_half,    X_test)),
    rmse_f(y_test, predict(f_sqrt,    X_test)),
    rmse_f(y_test, predict(f_rnd_t,   X_test)),
    rmse_f(y_test, predict(f_rnd_f,   X_test))
  ), 3)
)
print(tbl, row.names = FALSE)

## ----categorical, eval=FALSE--------------------------------------------------
# data(diamonds, package = "ggplot2")
# set.seed(7)
# idx   <- sample(nrow(diamonds), 2000)
# X_dia <- diamonds[idx, c("carat","cut","color","clarity",
#                           "depth","table","x","y","z")]
# y_dia <- diamonds$price[idx]
# 
# forest_dia <- SWORD(X_dia, y_dia, m = 50, oob = TRUE, verbose = FALSE)
# print(forest_dia)
# 
# # VI grouped by original variable (not by dummy column)
# VI_SWORD(forest_dia)
# 
# # PDP for an ordered factor
# pdp_sword(forest_dia, X_dia, "cut")

## ----formula, eval=FALSE------------------------------------------------------
# df <- cbind(X_train, medv = y_train)
# 
# tree2   <- TORS(medv ~ ., data = df, nmin = 15, cp = 0.02)
# forest2 <- SWORD(medv ~ ., data = df, m = 50, verbose = FALSE)
# predict(forest2, Boston[test, ])


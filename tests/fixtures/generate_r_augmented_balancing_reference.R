#!/usr/bin/env Rscript

# Generate estimator-parity fixtures with base R and quadprog. The objective
# and constraints match sdid_unit.R and sdid_time.R in the source project.

if (!requireNamespace("quadprog", quietly = TRUE)) {
  stop("The quadprog package is required to regenerate this fixture.")
}

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- sub("^--file=", "", script_argument[[1]])
fixture_directory <- dirname(normalizePath(script_file, mustWork = TRUE))

set.seed(4301)
n_control <- 8
n_treated <- 3
t_pre <- 10
t_post <- 4
n_periods <- t_pre + t_post
time <- seq_len(n_periods)
factors <- rbind(seq(-1, 1, length.out = n_periods), sin(time * pi / 7))
control_loadings <- matrix(rnorm(n_control * 2), nrow = n_control)
controls <- control_loadings %*% factors + matrix(rnorm(n_control * n_periods, sd = 0.08),
                                                  nrow = n_control)
true_weights <- rbind(
  c(0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.15, 0.10),
  c(0.15, 0.05, 0.10, 0.10, 0.20, 0.10, 0.15, 0.15),
  c(0.10, 0.15, 0.05, 0.15, 0.10, 0.20, 0.10, 0.15)
)
treated_untreated <- true_weights %*% controls
treated_untreated <- treated_untreated + c(-0.2, 0.0, 0.2)
effect_path <- c(rep(0, t_pre), seq(0.3, 1.2, length.out = t_post))
y <- rbind(controls, treated_untreated + effect_path)
w <- matrix(0, nrow = n_control + n_treated, ncol = n_periods)
w[(n_control + 1):(n_control + n_treated), (t_pre + 1):n_periods] <- 1
outcome_model <- outer(seq_len(nrow(y)), seq_len(ncol(y)),
                       function(unit, period) 0.04 * unit + 0.03 * period)

panel <- data.frame(
  unit = rep(seq_len(nrow(y)) - 1, each = n_periods),
  period = rep(seq_len(n_periods) - 1, nrow(y)),
  outcome = c(t(y)),
  treatment = c(t(w)),
  outcome_model = c(t(outcome_model))
)
write.csv(panel, file.path(fixture_directory, "r_augmented_balancing_panel.csv"),
          row.names = FALSE)

simplex_ridge <- function(design, target, zeta) {
  centered_design <- sweep(design, 2, colMeans(design), "-")
  centered_target <- target - mean(target)
  penalty <- zeta^2 * nrow(design)
  quadratic <- 2 * (crossprod(centered_design) + penalty * diag(ncol(design)))
  linear <- 2 * crossprod(centered_design, centered_target)
  constraints <- cbind(rep(1, ncol(design)), diag(ncol(design)))
  bounds <- c(1, rep(0, ncol(design)))
  quadprog::solve.QP(quadratic, linear, constraints, bounds, meq = 1)$solution
}

fit_case <- function(balance, unit_target, time_target, balance_on) {
  residual <- y - outcome_model
  balancing <- if (balance_on == "raw") y else residual
  control_pre <- balancing[seq_len(n_control), seq_len(t_pre), drop = FALSE]
  sigma <- sd(as.vector(control_pre[, -1, drop = FALSE] -
                        control_pre[, -t_pre, drop = FALSE]))
  target_size <- if (unit_target == "cohort") n_treated else 1
  zeta_omega <- (target_size * t_post)^0.25 * sigma
  zeta_lambda <- 1e-6 * sigma

  if (balance %in% c("unit", "double")) {
    if (unit_target == "cohort") {
      fitted <- simplex_ridge(t(control_pre),
                              colMeans(balancing[(n_control + 1):nrow(y),
                                                 seq_len(t_pre), drop = FALSE]),
                              zeta_omega)
      omega <- matrix(rep(fitted, n_treated), nrow = n_control)
    } else {
      omega <- vapply(seq_len(n_treated), function(index) {
        simplex_ridge(t(control_pre),
                      balancing[n_control + index, seq_len(t_pre)], zeta_omega)
      }, numeric(n_control))
    }
  } else {
    omega <- matrix(1 / n_control, nrow = n_control, ncol = n_treated)
  }

  if (balance %in% c("time", "double")) {
    control_post <- balancing[seq_len(n_control), (t_pre + 1):n_periods, drop = FALSE]
    if (time_target == "all") {
      fitted <- simplex_ridge(control_pre, rowMeans(control_post), zeta_lambda)
      lambda <- matrix(rep(fitted, t_post), nrow = t_pre)
    } else {
      lambda <- vapply(seq_len(t_post), function(index) {
        simplex_ridge(control_pre, control_post[, index], zeta_lambda)
      }, numeric(t_pre))
    }
  } else {
    lambda <- matrix(1 / t_pre, nrow = t_pre, ncol = t_post)
  }

  effects <- matrix(NA_real_, nrow = n_treated, ncol = t_post)
  control_residual <- residual[seq_len(n_control), , drop = FALSE]
  for (treated_index in seq_len(n_treated)) {
    unit <- n_control + treated_index
    for (post_index in seq_len(t_post)) {
      period <- t_pre + post_index
      predicted <- outcome_model[unit, period] +
        sum(omega[, treated_index] * control_residual[, period]) +
        sum(lambda[, post_index] * residual[unit, seq_len(t_pre)]) -
        as.numeric(t(lambda[, post_index]) %*%
                   t(control_residual[, seq_len(t_pre), drop = FALSE]) %*%
                   omega[, treated_index])
      effects[treated_index, post_index] <- y[unit, period] - predicted
    }
  }
  data.frame(
    balance = balance,
    unit_target = unit_target,
    time_target = time_target,
    balance_on = balance_on,
    zeta_omega = zeta_omega,
    zeta_lambda = zeta_lambda,
    att = mean(effects)
  )
}

grid <- expand.grid(
  balance = c("unit", "time", "double"),
  unit_target = c("cohort", "individual"),
  time_target = c("all", "period"),
  balance_on = c("raw", "residual"),
  stringsAsFactors = FALSE
)
results <- do.call(rbind, lapply(seq_len(nrow(grid)), function(index) {
  fit_case(grid$balance[index], grid$unit_target[index], grid$time_target[index],
           grid$balance_on[index])
}))
write.csv(results, file.path(fixture_directory, "r_augmented_balancing_results.csv"),
          row.names = FALSE)
message("Wrote ", nrow(results), " R estimator reference cases.")

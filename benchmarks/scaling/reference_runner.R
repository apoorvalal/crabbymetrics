#!/usr/bin/env Rscript

# Run one R reference cell. Unsupported or unavailable references are explicit,
# rather than silently substituting a semantically different estimator.

suppressWarnings(suppressMessages(library(jsonlite)))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) stop("expected estimator, implementation, n, k, seed")
estimator <- args[[1]]
implementation <- args[[2]]
n <- as.integer(args[[3]])
k <- as.integer(args[[4]])
seed <- as.integer(args[[5]])
set.seed(seed)

emit <- function(status, elapsed = NA_real_, checksum = NA_real_, error = NULL) {
  payload <- list(
    estimator = estimator,
    implementation = implementation,
    n = n,
    k = k,
    status = status,
    fit_seconds = elapsed,
    checksum = checksum,
    library_version = R.version.string
  )
  if (!is.null(error)) payload$error <- error
  cat(toJSON(payload, auto_unbox = TRUE, na = "null"), "\n")
}

require_or_emit <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    emit("missing_dependency", error = paste("R package not installed:", package))
    quit(status = 1)
  }
}

x <- matrix(rnorm(n * k), nrow = n, ncol = k)
beta <- seq(0.2, 1.0, length.out = k) / sqrt(k)
y <- as.vector(x %*% beta + rnorm(n))

tryCatch({
  started <- proc.time()[["elapsed"]]
  if (implementation == "r-fixest-feols") {
    require_or_emit("fixest")
    frame <- as.data.frame(x)
    names(frame) <- paste0("x", seq_len(k))
    frame$y <- y
    frame$fe <- rep(seq_len(max(2L, min(n %/% 20L, 1000L))), length.out = n)
    rhs <- paste(names(frame)[seq_len(k)], collapse = "+")
    model <- fixest::feols(as.formula(paste0("y~", rhs, "|fe")), data = frame, vcov = "iid")
    checksum <- sum(stats::coef(model))
  } else if (implementation == "r-survival-coxph") {
    require_or_emit("survival")
    tt <- rexp(n, exp(pmax(-1, pmin(1, as.vector(x %*% beta))))) + 0.01
    event <- rbinom(n, 1, 0.8)
    frame <- data.frame(time = tt, event = event, x)
    model <- survival::coxph(survival::Surv(time, event) ~ ., data = frame, ties = "breslow")
    checksum <- sum(stats::coef(model))
  } else if (implementation == "r-survival-andersen-gill") {
    require_or_emit("survival")
    start <- runif(n)
    stop <- start + rexp(n) + 0.01
    event <- rbinom(n, 1, 0.7)
    frame <- data.frame(start = start, stop = stop, event = event, x)
    model <- survival::coxph(survival::Surv(start, stop, event) ~ ., data = frame, ties = "breslow")
    checksum <- sum(stats::coef(model))
  } else {
    stop("reference is registered for provenance but has no semantically exact generic-grid runner")
  }
  emit("ok", proc.time()[["elapsed"]] - started, checksum)
}, error = function(exc) {
  emit("error", error = paste(class(exc)[[1]], conditionMessage(exc), sep = ": "))
  quit(status = 1)
})

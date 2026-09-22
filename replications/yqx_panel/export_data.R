args <- commandArgs(trailingOnly=TRUE)
root <- args[[1]]
for (f in list.files(root, pattern="\\.(rda|RData)$", full.names=TRUE)) {
  e <- new.env()
  for (name in load(f, envir=e)) {
    x <- get(name, envir=e)
    if (is.data.frame(x)) write.csv(x, file.path(root, paste0(name,".csv")), row.names=FALSE)
  }
}

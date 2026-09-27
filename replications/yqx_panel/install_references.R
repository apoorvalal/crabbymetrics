# Local-only reference installation, using recorded versions.
args<-commandArgs(trailingOnly=TRUE);root<-normalizePath(args[[1]])
lib<-file.path(root,'.cache','R-library');dir.create(lib,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(lib,.libPaths()));options(repos=c(CRAN='https://cloud.r-project.org'))
if(!requireNamespace('remotes',quietly=TRUE))install.packages('remotes',lib=lib)
pins<-read.csv(file.path(root,'results','reference-packages.csv'),stringsAsFactors=FALSE,na.strings=character())
for(i in seq_len(nrow(pins))) {
  p<-pins[i,]
  if(requireNamespace(p$package,quietly=TRUE) && as.character(packageVersion(p$package))==p$version)next
  if(nchar(p$remote_sha)>0)remotes::install_github(paste0(p$remote_repo,'@',p$remote_sha),lib=lib,upgrade='never',dependencies=NA)
  else remotes::install_version(p$package,version=p$version,lib=lib,upgrade='never',dependencies=NA)
}

args<-commandArgs(trailingOnly=TRUE);root<-normalizePath(args[[1]])
.libPaths(c(file.path(root,'.cache','R-library'),.libPaths()))
library(fixest)
d<-read.csv(file.path(root,'.cache','archive-Schafer_2022_AJPS.csv'))
rows<-lapply(c(1e-6,1e-10),function(t) {
  m<-feols(y~treatment_alt|fe1+fe2+fe3,d,fixef.tol=t,fixef.iter=100000,fixef.rm='none')
  data.frame(tolerance=t,coef=coef(m)[[1]])
})
write.csv(do.call(rbind,rows),file.path(root,'results','schafer-reference-tolerance.csv'),row.names=FALSE)

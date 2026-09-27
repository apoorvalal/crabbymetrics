# Independent package references for the exact exported regression samples.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[[1]])
.libPaths(c(file.path(root,'.cache','R-library'), .libPaths()))
suppressPackageStartupMessages(library(fixest))
jobs <- jsonlite::fromJSON(file.path(root,'.cache','linear-jobs.json'),simplifyVector=FALSE)
out <- list()
for (j in jobs) {
  d <- read.csv(file.path(root,'.cache',paste0('design-',j$name,'.csv')))
  f <- as.formula(j$formula)
  cl <- if (is.null(j$cluster)) NULL else as.formula(paste0('~',j$cluster))
  m <- feols(f,d,vcov=if(is.null(cl)) 'hetero' else cl,notes=FALSE)
  s <- summary(m)
  # Account for all absorbed FE, matching the current CrabbyMetrics CR1 df.
  full <- summary(m,ssc=ssc(adj=TRUE,fixef.K='full',cluster.adj=TRUE))
  for (term in unlist(j$keep)) {
    if (term %in% names(coef(m))) out[[length(out)+1]] <- data.frame(
      model=j$name,term=term,reference=coef(m)[[term]],reference_se=se(s)[[term]],
      aligned_se=se(full)[[term]],n=nobs(m),reference_package='fixest')
  }
}
write.csv(do.call(rbind,out),file.path(root,'results','linear-reference.csv'),row.names=FALSE)

# Reproduce plm's exact HC1 convention from the lecture script, including RE.
if(requireNamespace('plm',quietly=TRUE)) {
  d <- read.csv(file.path(root,'.cache','fatalities.csv'))
  d$fatal_rate <- 10000*d$fatal/d$pop
  p <- plm::pdata.frame(d,index=c('state','year'))
  fits <- list(pooled=plm::plm(fatal_rate~beertax,p,model='pooling'),
               unit_fe=plm::plm(fatal_rate~beertax,p,model='within'),
               re=plm::plm(fatal_rate~beertax,p,model='random',random.method='swar'),
               twfe=plm::plm(fatal_rate~beertax,p,model='within',effect='twoways'))
  z <- do.call(rbind,lapply(names(fits),function(n) {
    m<-fits[[n]]; V<-plm::vcovHC(m,method='arellano',type='HC1',cluster='group')
    data.frame(model=paste0('fatalities_',n),estimate=coef(m)[['beertax']],se=sqrt(V['beertax','beertax']))
  }))
  write.csv(z,file.path(root,'results','fatalities-plm.csv'),row.names=FALSE)
}
writeLines(capture.output(sessionInfo()),file.path(root,'results','R-session.txt'))

# Independent empirical references; run one family per invocation for fault isolation.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[[1]])
family <- args[[2]]
.libPaths(c(file.path(root,'.cache','R-library'), .libPaths()))
set.seed(20260921)
options(warn=1)
read <- function(n) read.csv(file.path(root,'.cache',paste0(n,'.csv')))
save <- function(x,n) write.csv(x,file.path(root,'results',paste0(n,'.csv')),row.names=FALSE)
store <- function(x,n) saveRDS(x,file.path(root,'.cache',paste0(n,'.rds')))

if(family=='fdid') {
  d <- read('mortality'); d$G <- as.integer(d$pczupu>0)
  X <- c('avggrain','nograin','urban','dis_bj','dis_pc','rice','minority','edu','lnpop')
  s <- fdid::fdid_prepare(d,'mortality',X,'G','countyid','year')
  all <- list(); dynamics <- list()
  for(method in c('did','ols1','ols2','ipw','aipw')) {
    set.seed(20260921)
    cat('FDID',method,'\n')
    cached <- file.path(root,'.cache',paste0('fdid-',method,'.rds'))
    fit <- if(file.exists(cached)) readRDS(cached) else fdid::fdid(s,tr_period=1958:1961,ref_period=1957,method=method,target.pop='all')
    store(fit,paste0('fdid-',method))
    all[[method]] <- data.frame(method=method,period='event',fit$est$event)
    dynamics[[method]] <- data.frame(method=method,year=1954:1966,fit$dynamic)
    save(do.call(rbind,all),'fdid-reference'); save(do.call(rbind,dynamics),'fdid-dynamic-reference')
  }
  f <- fdid::fdid(s,tr_period=1958:1961,ref_period=1957,method='ebal',target.pop='1')
  save(data.frame(method='ebal',target='group-1 (not all)',f$est$event),'fdid-ebal-reference')
  # The lecture's multivalued example: no, below-positive-median, above-positive-median genealogy.
  cut <- median(d$pczupu[d$year==1957 & d$pczupu>0]); d$G3 <- ifelse(d$pczupu==0,0,ifelse(d$pczupu<cut,1,2))
  z <- list()
  for(g in 1:2) {
    set.seed(20260921+g)
    a <- d[d$G3 %in% c(0,g),]; a$G <- as.integer(a$G3==g)
    s <- fdid::fdid_prepare(a,'mortality',X,'G','countyid','year')
    f <- fdid::fdid(s,tr_period=1958:1961,ref_period=1957,method='aipw',target.pop='all')
    z[[g]] <- data.frame(comparison=paste0(g,'-vs-0'),n=length(unique(a$countyid)),f$est$event)
  }
  save(do.call(rbind,z),'fdid-multivalued-reference')
}
if(family=='divorce') {
  d <- read('divorce'); d <- subset(d,sex==2 & year>=1964 & year<=1996)
  d$asmrs <- d$suiciderate_jag*1e6; d$post <- d$unilateral
  b <- bacondecomp::bacon(asmrs~post,data=d[,c('st','year','asmrs','post')],id_var='st',time_var='year',quietly=TRUE)
  save(b,'bacon-reference')
  save(data.frame(estimate=sum(b$weight*b$estimate),weight_sum=sum(b$weight)),'bacon-total')
  d <- subset(d,divyear>1964); d$rel_b <- pmax(-9,pmin(16,d$year-d$divyear))
  f <- fixest::feols(asmrs~i(rel_b,ref=-1)|st+year,d,cluster=~st)
  save(data.frame(term=names(coef(f)),estimate=coef(f),se=fixest::se(f)),'divorce-source-event')
}
if(family=='smoking') {
  d <- read('smoking'); d$treated <- as.integer(d$state=='California' & d$year>=1989)
  out <- list(); paths <- list()
  for(method in c('scm','ridge','ridge_covariates')) {
    form <- if(method=='ridge_covariates') cigsale~treated|lnincome+beer+age15to24+retprice else cigsale~treated
    f <- augsynth::augsynth(form,unit=state,time=year,data=d,progfunc=if(method=='scm') 'None' else 'Ridge',scm=TRUE,cov_agg=function(x)mean(x,na.rm=TRUE))
    s <- summary(f); store(f,paste0('augsynth-',method))
    out[[method]] <- data.frame(method=method,s$average_att)
    paths[[method]] <- data.frame(method=method,s$att)
    if(method=='scm') save(data.frame(state=rownames(f$weights),weight=as.numeric(f$weights)),'augsynth-scm-weights')
  }
  save(do.call(rbind,out),'augsynth-reference'); save(do.call(rbind,paths),'augsynth-paths')
  p <- synthdid::panel.matrices(d[,c('state','year','cigsale','treated')],unit='state',time='year',outcome='cigsale',treatment='treated')
  f <- synthdid::synthdid_estimate(p$Y,p$N0,p$T0); store(f,'synthdid')
  se <- sqrt(as.numeric(vcov(f,method='placebo',replications=100)))
  save(data.frame(estimate=as.numeric(f),placebo_se=se,draws=100,seed=20260921),'synthdid-reference')
  w <- attr(f,'weights'); save(data.frame(state=rownames(p$Y)[seq_len(p$N0)],weight=w$omega),'synthdid-unit-weights')
  save(data.frame(year=colnames(p$Y)[seq_len(p$T0)],weight=w$lambda),'synthdid-time-weights')
}
if(family=='fect') {
  out <- list()
  for(dataset in c('gs2020','turnout','hh2019')) {
    d <- read(dataset)
    formula <- switch(dataset,gs2020=general_sharetotal_A_all~cand_A_all,turnout=turnout~policy_edr+policy_mail_in+policy_motor,hh2019=nat_rate_ord~indirect)
    index <- switch(dataset,gs2020=c('district_final','cycle'),turnout=c('abb','year'),hh2019=c('bfs','year'))
    for(method in if(dataset=='turnout') c('fe','ife') else 'fe') {
      key <- paste(dataset,method,sep='_'); cat('FECT',key,'\n')
      cached <- file.path(root,'.cache',paste0(key,'.rds'))
      f <- if(file.exists(cached)) readRDS(cached) else fect::fect(formula,data=d,index=index,force='two-way',method=method,r=if(method=='ife') 2 else 0,CV=FALSE,se=FALSE,parallel=FALSE,na.rm=TRUE,seed=20260921)
      store(f,key); cat('names:',names(f),'\n'); cat('ATT:',f$att.avg,'\n')
      last <- function(n) f[[tail(which(names(f)==n),1)]]
      yy <- last('Y'); dd <- last('D'); xx <- last('X')
      out[[key]] <- data.frame(model=key,estimate=f$att.avg,treated_cells=sum(dd==1 & f$I==1,na.rm=TRUE))
      save(do.call(rbind,out),'fect-reference')
      # Export estimation masks, outcome surfaces, and ID order for identical-sample Python audits.
      for(n in c('Y','D','I','Y.ct','Y.dat','D.dat','I.dat','Y0','X')) {
        if(!n %in% names(f)) next
        a <- last(n)
        if(is.matrix(a)) write.csv(a,file.path(root,'.cache',paste0(key,'-',gsub('\\.','_',n),'.csv')),row.names=FALSE)
      }
      grid <- expand.grid(time=seq_len(nrow(yy)),unit=seq_len(ncol(yy)))
      grid$y <- as.vector(yy); grid$d <- as.vector(dd); grid$observed <- as.vector(f$I)
      grid$ref_counterfactual <- as.vector(f$Y.ct)
      if(length(xx)>0) for(j in seq_len(dim(xx)[3])) grid[[paste0('x',j)]] <- as.vector(xx[,,j])
      write.csv(grid,file.path(root,'.cache',paste0(key,'-grid.csv')),row.names=FALSE)
    }
  }
}
writeLines(capture.output(sessionInfo()),file.path(root,'results',paste0('R-session-',family,'.txt')))

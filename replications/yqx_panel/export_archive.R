# Source-derived standardized TWFE designs for all 49 reanalyses.
args<-commandArgs(trailingOnly=TRUE);root<-normalizePath(args[[1]])
.libPaths(c(file.path(root,'.cache','R-library'),.libPaths()))
suppressPackageStartupMessages(library(fixest))
jobs<-jsonlite::fromJSON(file.path(root,'.cache','archive-jobs.json'),simplifyVector=FALSE)
setwd(file.path(root,'.cache','replication'))
load('output/summary.RData'); published<-d
write.csv(published,file.path(root,'results','reanalysis-published-summary.csv'),row.names=FALSE)
year<-function(x)as.integer(format(x,'%Y'));month<-function(x)as.integer(format(x,'%m'))
out<-list(); specs<-list()
for(j in jobs) {
  cat(j$name,'\n')
  # These two datasets contain 89m and 158m observations in the metadata.
  # Do not decompress them into R on a 16-GiB workstation.
  if(grepl('^(Hall_Yoder|Sanford)_',j$name)) {
    specs[[j$name]]<-list(name=j$name,status='resource-limited: published n > 80 million; not loaded')
    next
  }
  tryCatch({
    e<-new.env(parent=globalenv());eval(parse(text=j$setup),e)
    df<-e$df;y<-e$y;tr<-e$d;fx<-unique(c(e$f,unlist(j$extra_effects)));cluster<-e$f[[1]]
    controls<-e$controls
    if(length(cluster)>1)cluster<-cluster[[1]] # standardized comparison uses unit clustering
    df<-df[complete.cases(df[,c(tr,fx)]),]
    means<-ave(df[[tr]],df[[fx[[1]]]],FUN=mean)
    df<-df[means<1,]
    cols<-unique(c(y,tr,controls,fx,cluster));df<-df[complete.cases(df[,cols]),cols,drop=FALSE]
    form<-as.formula(paste(y,'~',paste(c(tr,controls),collapse='+'),'|',paste(fx,collapse='+')))
    m<-feols(form,df,cluster=as.formula(paste0('~',cluster)),fixef.rm='none',notes=FALSE)
    rhs<-model.matrix(m,type='rhs'); used<-colnames(rhs)
    # Drop only regressors the R rank check removed. Record them explicitly.
    rhs<-rhs[,names(coef(m)),drop=FALSE];used<-colnames(rhs)
    fe<-as.data.frame(m$fixef_id);names(fe)<-paste0('fe',seq_along(fe))
    if(nrow(rhs)!=nrow(df))stop('Unexpected sample removal; explicit audit needed')
    design<-cbind(data.frame(y=df[[y]],cluster=as.integer(factor(df[[cluster]]))),as.data.frame(rhs),fe)
    write.csv(design,file.path(root,'.cache',paste0('archive-',j$name,'.csv')),row.names=FALSE)
    s<-summary(m,ssc=ssc(adj=TRUE,fixef.K='full',cluster.adj=TRUE))
    out[[j$name]]<-data.frame(name=j$name,term=tr,estimate=coef(m)[[tr]],se=se(m)[[tr]],aligned_se=se(s)[[tr]],n=nobs(m))
    specs[[j$name]]<-list(name=j$name,status='prepared',treatment=tr,regressors=used,effects=names(fe),formula=deparse(form),removed=m$collin.var,n=nobs(m),
      scope=if(isTRUE(j$unit_trends)) 'baseline only: source also requires unit-specific slopes' else 'source point specification')
  },error=function(err){specs[[j$name]]<<-list(name=j$name,status=paste('error:',conditionMessage(err)))})
  gc()
  write.csv(do.call(rbind,out),file.path(root,'results','archive-reference.csv'),row.names=FALSE)
  jsonlite::write_json(unname(specs),file.path(root,'results','archive-specifications.json'),pretty=TRUE,auto_unbox=TRUE,null='null')
}

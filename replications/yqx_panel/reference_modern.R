# Lecture-facing imputation applications on the archived author data.
args<-commandArgs(trailingOnly=TRUE); root<-normalizePath(args[[1]])
.libPaths(c(file.path(root,'.cache','R-library'),.libPaths()))
jobs<-jsonlite::fromJSON(file.path(root,'.cache','archive-jobs.json'),simplifyVector=FALSE)
setwd(file.path(root,'.cache','replication'))
selected<-c('Bischof_Wagner_2019_AJPS','Grumbach_Sahn_2020_APSR','Trounstine_2020_APSR','Christensen_Garfias_2021_JOP')
rows<-list()
for(j in jobs) if(j$name %in% selected) {
  e<-new.env(); eval(parse(text=j$setup),e)
  variants<-if(grepl('Christensen',j$name))c('full','trimmed') else 'full'
  for(variant in variants) {
    df<-e$df
    if(variant=='trimmed')df<-df[which(df$anyswitch_period==1 & df$year<=2014),]
    key<-paste0('modern_',j$name,'_',variant);cat(key,'\n')
    fit<-fect::fect(data=df,Y=e$y,D=e$d,X=e$controls,index=e$f,method='fe',force='two-way',na.rm=TRUE,se=FALSE,parallel=FALSE,seed=20260921)
    saveRDS(fit,file.path(root,'.cache',paste0(key,'.rds')))
    get<-function(n) fit[[tail(which(names(fit)==n),1)]]
    yy<-get('Y');dd<-get('D');xx<-get('X')
    rows[[key]]<-data.frame(model=key,estimate=fit$att.avg,treated_cells=sum(dd==1 & fit$I==1,na.rm=TRUE),n_units=ncol(yy),n_periods=nrow(yy))
    write.csv(do.call(rbind,rows),file.path(root,'results','modern-fect-reference.csv'),row.names=FALSE)
    if(grepl('Bischof|Grumbach',j$name)) {
      grid<-expand.grid(time=seq_len(nrow(yy)),unit=seq_len(ncol(yy)))
      grid$y<-as.vector(yy);grid$d<-as.vector(dd);grid$observed<-as.vector(fit$I);grid$ref_counterfactual<-as.vector(fit$Y.ct)
      if(length(xx)>0)for(k in seq_len(dim(xx)[3]))grid[[paste0('x',k)]]<-as.vector(xx[,,k])
      write.csv(grid,file.path(root,'.cache',paste0(key,'-grid.csv')),row.names=FALSE)
    }
  }
}

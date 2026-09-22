# Canonical ADH predictor recipe reconstructed on the public smoking panel.
# The lecture ships figures but not its ca1--ca8 script: this is not pixel-exact provenance.
args<-commandArgs(trailingOnly=TRUE);root<-normalizePath(args[[1]])
.libPaths(c(file.path(root,'.cache','R-library'),.libPaths()))
d<-read.csv(file.path(root,'.cache','smoking.csv'));d$id<-as.numeric(factor(d$state))
states<-unique(d[,c('id','state')]);states<-states[order(states$id),]
all<-list();balance<-NULL
for(id in states$id) {
 cat('Synth',states$state[states$id==id],'\n')
 dp<-Synth::dataprep(foo=d,predictors=c('lnincome','retprice','age15to24'),predictors.op='mean',
   time.predictors.prior=1980:1988,special.predictors=list(list('beer',1984:1988,'mean'),list('cigsale',1975,'mean'),list('cigsale',1980,'mean'),list('cigsale',1988,'mean')),
   dependent='cigsale',unit.variable='id',unit.names.variable='state',time.variable='year',
   treatment.identifier=id,controls.identifier=setdiff(states$id,id),time.optimize.ssr=1970:1988,time.plot=1970:2000)
 fit<-Synth::synth(dp,verbose=FALSE)
 gap<-as.vector(dp$Y1plot-dp$Y0plot%*%fit$solution.w)
 all[[as.character(id)]]<-data.frame(state=states$state[states$id==id],year=1970:2000,gap=gap,pre_mspe=mean(gap[1:19]^2),post_mspe=mean(gap[20:31]^2))
 write.csv(do.call(rbind,all),file.path(root,'results','synth-canonical-placebos.csv'),row.names=FALSE)
 if(states$state[states$id==id]=='California') {
   tab<-Synth::synth.tab(dataprep.res=dp,synth.res=fit)
   write.csv(tab$tab.pred,file.path(root,'results','synth-canonical-balance.csv'))
   write.csv(tab$tab.w,file.path(root,'results','synth-canonical-weights.csv'),row.names=FALSE)
   # An explicitly R-assisted inner-problem test for the missing predictor-V API.
   features<-cbind(treated=as.vector(dp$X1),dp$X0)
   features<-features*sqrt(as.numeric(unlist(fit$solution.v)))/apply(cbind(dp$X1,dp$X0),1,sd)
   write.csv(features,file.path(root,'.cache','synth-weighted-features.csv'),row.names=FALSE)
 }
}

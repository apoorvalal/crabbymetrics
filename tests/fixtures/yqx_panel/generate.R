# Original deterministic fixtures: no third-party microdata.
# Regenerate from repository root: Rscript tests/fixtures/yqx_panel/generate.R
root <- 'tests/fixtures/yqx_panel'
options(digits=17)
write <- function(x,n) write.csv(x,file.path(root,n),row.names=FALSE,na='NA')
d <- expand.grid(unit=0:11, time=0:7)
d <- d[order(d$unit,d$time),]
d$x1 <- sin((d$unit+1)*(d$time+1)*.27)
d$x2 <- cos((d$unit+2)*(d$time+1)*.16)
d$treatment <- as.numeric(d$unit>=5 & d$time>=3 & !((d$unit%%2==1) & (d$time%%3==0)))
d$observed <- as.integer(!((d$unit+d$time*3)%%17==0))
d$weight <- 1+(d$unit+d$time)%%4/3
d$y <- 4+sin(d$unit*.5)+d$time*.3+.7*d$x1-.35*d$x2+
    .15*cos(d$unit*.3+d$time*.7)+d$treatment*(1.1+.08*d$unit-.05*d$time)
d$unit <- factor(d$unit);d$time <- factor(d$time)
all <- subset(d,observed==1)
cf <- list()
for (weighted in c(FALSE,TRUE)) {
    z <- subset(all,treatment==0)
    m <- lm(y~x1+x2+unit+time,data=z,weights=if(weighted) z$weight else rep(1,nrow(z)))
    pred <- predict(m,newdata=d)
    treated <- all$treatment==1
    p <- predict(m,newdata=all)
    att <- weighted.mean(all$y[treated]-p[treated], if(weighted) all$weight[treated] else rep(1,sum(treated)))
    cf[[length(cf)+1]] <- data.frame(weighted=weighted,att=att,beta1=coef(m)['x1'],beta2=coef(m)['x2'])
    d[[if(weighted) 'cf_weighted' else 'cf']] <- pred
}
write(do.call(rbind,cf),'imputation-reference.csv')
# Fully saturated lm, with cluster score meat; independent of within absorption.
m <- lm(y~x1+x2+unit+time,data=all)
X <- model.matrix(m); e <- resid(m); bread <- solve(crossprod(X))
scores <- X*as.numeric(e); sums <- rowsum(scores,all$unit)
base <- bread %*% crossprod(sums) %*% bread
n <- nrow(X); g <- nrow(sums); k <- ncol(X)
refs <- list()
for (policy in c('full','non_nested','none')) {
    K <- switch(policy, full=k, non_nested=2+nlevels(all$time), none=2)
    for (ssc in c('cluster','hc1','none')) for (gc in c(TRUE,FALSE)) {
        size <- switch(ssc,cluster=(n-1)/(n-K),hc1=n/(n-K),none=1)
        cov <- base[2:3,2:3]*size*if(gc) g/(g-1) else 1
        refs[[length(refs)+1]] <- data.frame(fe_df=policy,ssc=ssc,cluster_correction=gc,beta1=coef(m)[2],beta2=coef(m)[3],v11=cov[1,1],v12=cov[1,2],v22=cov[2,2])
    }
}
write(do.call(rbind,refs),'fe-reference.csv')
d$unit <- as.integer(as.character(d$unit)); d$time <- as.integer(as.character(d$time))
write(d,'panel.csv')
# Exhaust all nonempty donor faces; base R equality-constrained least squares.
T <- 28; J <- 6; tt <- 0:(T-1)
D <- sapply(1:J,function(j) 100+20*sin(tt*.12+j*.4)+j*tt*.15+5*cos(tt*.7+j))
y <- .6*D[,2]+.4*D[,5]+2*sin(tt*.8)-.8*tt
y[tt>=18] <- y[tt>=18]-14
A <- D[tt<18,]; b <- y[tt<18]
best <- Inf; weights <- NULL
for (mask in 1:(2^J-1)) {
    ids <- which(as.logical(intToBits(mask)[1:J])); h <- crossprod(A[,ids,drop=FALSE]); l <- crossprod(A[,ids,drop=FALSE],b)
    K <- rbind(cbind(h,1),c(rep(1,length(ids)),0))
    sol <- tryCatch(solve(K,c(l,1)),error=function(e) NULL)
    if(is.null(sol) || min(sol[seq_along(ids)]) < -1e-10) next
    w <- rep(0,J); w[ids] <- pmax(sol[seq_along(ids)],0); w <- w/sum(w)
    loss <- mean((A%*%w-b)^2)
    if(loss<best) {best <- loss; weights <- w}
}
stopifnot(!is.null(weights),sum(weights==0)>0)
write(data.frame(time=tt,treated=y,D),'scm.csv')
write(data.frame(donor=1:J,weight=weights),'scm-weights.csv')
write(data.frame(pre_rmse=sqrt(best),att=mean((y-D%*%weights)[tt>=18])),'scm-reference.csv')
writeLines(c('Original generated data; base R lm and exhaustive simplex-face optimization.',R.version.string), file.path(root,'provenance.txt'))

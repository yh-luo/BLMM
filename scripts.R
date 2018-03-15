dat = read.table("data/brain.dat")
colnames(dat) = c("period", "id", "treatment", "rating")
dat$treatment = as.factor(dat$treatment)
# reset id
dat$id = 1:length(unique(dat$id))
# relative time
dat$time = as.integer(dat$period-1)

pacman::p_load(lme4, tidyverse, rstan, coda)
ggplot(dat, aes(x=time, y=rating))+
  geom_point(size=3)+
  geom_line(aes(group=id))+
  stat_smooth(method="lm", size=3)+
  scale_x_continuous(labels=c(0,1,2), breaks=seq(0,2))+
  labs(x="\nTime", y="Rating\n")+
  theme_classic()+
  theme(axis.title=element_text(color="black", size=12),
        axis.text=element_text(color="black", size=12))

rst = lmer(rating~time+(1+time|id), data=dat)
summary(rst)


# bayesian
stanDat = list(time = as.integer(dat$time),
               id = as.integer(dat$id),
               rating = dat$rating,
               N = nrow(dat),  # number of data points
               J = length(unique(dat$id)),# number of subjects
               K = length(unique(dat$time)))  # number of waves
ranSlpFit = stan(file="ranIntSlp.stan", data=stanDat, iter=2000, chains=4)

print(ranSlpFit, pars = c("beta", "sigma_e", "sigma_u"),
      probs = c(0.025, 0.5, 0.975))

beta1 <- extract(ranSlpFit, pars = c("beta[2]"))$beta
# 95CI
print(signif(quantile(beta1, probs = c(0.025, 0.5, 0.975)), 3))
mean(beta1)
mean(beta1 < 0)

# L matrices
L_u <- extract(ranSlpFit, pars = "L_u")$L_u
# correlation parameters
cor_u <- apply(L_u, 1, function(x) tcrossprod(x)[1, 2])
print(signif(quantile(cor_u, probs = c(0.025, 0.5, 0.975)), 2))
print(mean(cor_u))

J<-length(unique(dat$id))
u<-matrix(nrow=2,ncol=J)
for(j in 1:J)
  for(i in 1:2)
    u[i,j]<-mean(extract(ranSlpFit,pars=c(paste("u[",i,",",j,"]",sep="")))[[1]])

N_sample<-length(extract(ranSlpFit,pars="L_u[1,1]")[[1]])
L_u<-array(dim=c(2,2,N_sample))
for(i in 1:2)
  for(j in 1:2)
    L_u[i,j,]<-extract(ranSlpFit,pars=c(paste("L_u[",i,",",j,"]",sep="")))[[1]]

rho_u<-numeric()
for(i in 1:N_sample){
  rho_u<-L_u[,,i]%*%t(L_u[,,i])
  rho_u[i]<-rho_u[1,2]
}
# Visualize the posterior distribution for the intercept beta[1] ...
plot(u[1,],u[2,], bg="black", xlab=expression(hat(u[0])),ylab=expression(hat(u[1])))
hist(rho_u,freq=FALSE,col="black",border="white",
     main=NULL,xlab=expression(hat(rho)[u]))

# Get HPD interval for beta[2]
beta1<-as.mcmc(unlist(extract(ranSlpFit,pars="beta[2]")))
betaHPD<-HPDinterval(beta1,prob=0.95)
# Get HPD interval for rho_u
N_iter<-length(beta1)
rho_u<-numeric(N_iter)
L_u<-array(dim=c(2,2,N_iter))
for(i in 1:2)
  for(j in 1:2)
    L_u[i,j,]<-extract(ranSlpFit,pars=paste("L_u[",i,",",j,"]",sep=""))[[1]]
for(i in 1:N_iter)
  rho_u[i] <- tcrossprod(L_u[,,i])[1,2]
rho_u<-as.mcmc(rho_u)
rhoHPD<-HPDinterval(rho_u,prob=0.95)
# PLOT HPD INTERVALS ON THE MARGINAL POSTERIORS
par(mfrow=c(1,2))
hist(beta1,freq=FALSE,col="black",border="white",xaxt="n",
     main=NULL,xlab=expression(hat(beta)[1]))
abline(v=betaHPD,lty=2,lwd=2)
axis(1, at = seq(-.1,.1,length.out=5), labels = seq(-.1,.1,length.out=5))
hist(rho_u,freq=FALSE,col="black",border="white",
     main=NULL,xlab=expression(hat(rho)[u]),xlim=c(-1,1))
abline(v=rhoHPD,lty=2,lwd=2)

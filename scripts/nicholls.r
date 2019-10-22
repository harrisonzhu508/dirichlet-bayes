#L15 Bayes Methods 2019 - Normal-Dirichlet mixture
#fit to galaxy data

library(MCMCpack) #dinvgamma for transparency

###############################################
#MCMC fitting the galaxy data

y=scan("http://www.stats.ox.ac.uk/~nicholls/BayesMethods/galaxies.txt") #82 galaxy transverse velocities
nd=length(y)           #number of data points

#the priors for the means are not quite the same as those I use in the RJ-MCMC example
#because I want the priors to be conjugat I use inverse gamma for sigma^2 rather than
#gamma on sigma as I used before mu~N(20,10) as before, but sigma^2~IGamma(alpha0,beta0)
#the choice of alpha0 and beta0 is intended to give a similar prior for sigma

log.prior.param<-function(mu,sg,mu0=20,sigma0=10,alpha0=2,beta0=1/9) {
  a=sum(dnorm(mu,mean=mu0,sd=sigma0,log=TRUE))
  b=sum(log(dinvgamma(sg^2,shape=alpha0,scale=beta0)))
  return(a+b)
}

sigma0=10; mu0=20
alpha0=2; beta0=1/9 #IGamma sg^2 mean=beta/(alpha-1)=9 so sg~3

#choice of alpha based on simulations above
alpha=1

#initialise MCMC state - S is known given mu but it is convenient to maintain both
S=S.init=rep(1,nd)
mu=mu.init=mean(y)
sg=sg.init=sd(y)

#run parameters and output storage
J=10000; SS=10;  #this is actually pretty minimal just enough for plausible estimates
CL=matrix(NA,J/SS,nd); PL=matrix(NA,J/SS,3); TH=list();

#MCMC J steps each step sweeps mu, sigma and the cluster membership
for (j in 1:J) {

  nS=table(S) #the number of pts in each cluster
  K=max(S)    #the number of clusters

  #go through each cluster and update mu[k] and sg[k] given the cluster membership S 
  for (k in 1:K) {
    i.in.k=which(S==k); yk=y[i.in.k] #which data points in this cluster
    nk=nS[k]                         #how many pts in this cluster
    if (nk>0) {
      #sample mu[k] given mu[-k], sg and S
      sgk=sg[k]
      bm=sqrt(1/(nk/sgk^2+1/sigma0^2))
      am=bm^2*(sum(yk)/sgk^2+mu0/sigma0^2)
      mu[k]=rnorm(1,mean=am,sd=bm)
      
      #sample sg[k] given sg[-k], mu and S
      as=alpha0+nk/2
      bs=beta0+(1/2)*sum((yk-mu[k])^2)
      sg[k]=1/sqrt(rgamma(1,shape=as,rate=bs))
    }
  } 
  #go through each point y[i] and update its cluster membership S[i]
  for (i in 1:nd) {
    #the number of pts per cluster and the number of clusters may have changed 
    nS=table(S)
    K=max(S)

    #we need the weights with i removed, we called this n_k^{-i} in lectures
    nSp=nS; k=S[i]; nSp[k]=nSp[k]-1
 
   #p(y[i]|mu[k],sg[k])*n_k^{-i} for each k=1:K
    pold=dnorm(y[i],mean=mu,sd=sg)*nSp

    #if we move i to a new cluster we need mu and sg values for the new cluster
    mkn=rnorm(1,mean=mu0,sd=sigma0)
    sgn=1/sqrt(rgamma(1,shape=alpha0,rate=beta0))

    #alpha*p(y[i]|mu.new,sig.new)
    pnew=alpha*dnorm(y[i],mean=mkn,sd=sgn)

    #the new cluster is either one of the old ones or a new one, so the conditional dbn
    #for the cluster of y[i] is the vector of normalised probabilities for each choice
    pr=c(pnew,pold); pr=pr/sum(pr)
    
    #pick a new cluster
    kn=sample(0:K,1,prob=pr)
    if (kn==0) { #new cluster, append mu.new and sig.new to mu, sg 
      S[i]=K+1
      mu=c(mu,mkn)
      sg=c(sg,sgn)
    } else {     #put the pt in cluster kn
      S[i]=kn
    }

    #keeping the cluster labels packed (ie 1:K) is delicate
    #this was the hardest bit to get right
    if (nSp[k]==0 & kn!=k) {
      ib=which(S>k) 
      S[ib]=S[ib]-1 
      mu=mu[-k]; sg=sg[-k]
    }
  }  

  #collect samples from the MCMC every SS steps
  if (j%%SS==0) {
    #conditional on the number in each cluster we know the dbn of the cluster weights w 
    TH[[j/SS]]=list(w=rdirichlet(1,alpha+table(S)),mu=mu,sigma=sg)
    CL[j/SS,]=S
    PL[j/SS,]=c(length(mu),min(S),max(S))  #check we are keeping the cluster labels 'packed'
  }

}

###
#Superficial Output Analysis
effectiveSize(as.mcmc(CL))
library(lattice); xyplot(as.mcmc(PL[,1]))
###

###
#visualise co-occurence matrix p(i,j) = prob i,j in same cluster 
Nsamp=J/SS;
com=matrix(0,nd,nd)
for (i in 1:nd) {
  for (j in 1:nd) {
    for (m in 1:Nsamp) {
      com[i,j]=com[i,j]+(CL[m,i]==CL[m,j])
    }
  }
}
com=com/(J/SS)
image(com)
###


###
#posterior dbn over number of components - exactly the same as for RJ-MCMC
#pdf('DPmixtureGhistM.pdf',8,4)
hist(d<-PL[,1],breaks=0.5:(0.5+max(PL[,1])),main='',freq=FALSE,
     xlab='Number of components m in mixture',ylab='density',xlim=c(0,1+max(PL[,1])))
#dev.off()
table(d)/Nsamp

#loglkd including the weights (not used in MCMC)
log.lkd<-function(y,w,mu,sigma) {
  tot=0
  for (i in 1:length(y)) {
    tot=tot+log(sum(w*dnorm(y[i],mean=mu,sd=sigma)))
  }
  return(tot)
}

###
#compute posterior predictive distributions - weights (w) vs. means (mu) scater plot
#pdf('DPmixtureGscatterMUW.pdf',8,4)
L=200
x=seq(0,40,length.out=L)
den=matrix(0,max(d)-min(d)+1,L)
plot(c(),c(),xlim=c(0,50),ylim=c(0,1),xlab='MU, colored by number of clusters in state',ylab='W component weight')
for (k in 1:Nsamp) { 
  w=TH[[k]]$w; mu=TH[[k]]$mu; sigma=TH[[k]]$sigma
  ind=d[k]-min(d)+1
  points(mu,w,pch='.',col=ind)
  den[ind,]=den[ind,]+exp(apply(t(x),2,log.lkd,w,mu,sigma))
}
cden=den/as.vector(table(d))
mden=apply(den,2,sum)/Nsamp
#dev.off()

#plot posterior predictive distributions
#pdf('DPmixtureGppd.pdf',8,4)
#data
hist(y,breaks=x,freq=FALSE,main='',xlab='velocity',ylab='density')
#posterior predicitve mean
for (i in 2:5) {#dim(cden)[1]) {
  #posteior preditive dbn conditioned on "i" components
  lines(x,cden[i,],lwd=2,col=i-1)
}
lines(x,mden,lwd=2,col=1)
#dev.off()

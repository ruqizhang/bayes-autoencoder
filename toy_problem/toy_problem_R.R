library(ggplot2)
set.seed(2018)

#generate data
y = rnorm(100)
z = runif(100, -1, 1)

beta = c(1, -1)

loglambda = beta[1]* y + beta[2] * z
x = rpois(100, exp(loglambda))

#log likelihood function
logll = Vectorize(function(beta1, beta2){
  loglambda = beta1 * y + beta2 * z
  
  sum(dpois(x, exp(loglambda), log = TRUE))
})

#conjugate prior function from Eq 6
prior_form = function(beta1, beta2, covar, response, a0, tau = 1){
  linear_pred = covar %*% c(beta1, beta2)
  bfun = sum(exp(-linear_pred))
  
  reg_pred = sum(response * linear_pred)
  
  a0 * tau * (reg_pred - bfun)
}

#compute contour of log likelihood
betaseq = seq(-3, 3, length.out = 50)
ll_contour = outer(betaseq, betaseq, FUN = logll)

contour(betaseq, betaseq, ll_contour,nlevels = 3000, 
        xlab = "beta1", ylab = "beta2", 
        main = "Profile of Poisson LL")

mle_model = glm(x ~ y + z -1, family = "poisson") #basically works
points(coef(mle_model)[1], coef(mle_model)[2], pch = 20, col = "red")

prior = Vectorize(function(beta1, beta2){
  prior_form(beta1, beta2, cbind(y, z), rep(0, length(x)), 1)
  })

prior_contour = outer(betaseq, betaseq, FUN = prior)
contour(betaseq, betaseq, prior_contour,nlevels = 3000, 
        xlab = "beta1", ylab = "beta2", 
        main = "Profile of Prior")

contour(betaseq, betaseq, ll_contour + prior_contour, nlevels = 3000, 
        xlab = "beta1", ylab = "beta2", main = "Profile of Likelihood + Prior")
points(coef(mle_model)[1], coef(mle_model)[2], pch = 20, col = "red")


posterior = Vectorize(function(beta1, beta2){
  one_sample = function(){
    zcurr = runif(100, -1, 1)
    prior_form(beta1, beta2, cbind(y, zcurr), x/2, 2)
  }
  logprobs = replicate(100, one_sample())
  mean(logprobs)
})
posterior_contour = outer(betaseq, betaseq, FUN = posterior)
contour(betaseq, betaseq, posterior_contour,nlevels = 3000, 
        xlab = "beta1", ylab = "beta2", 
        main = "Profile of Posterior")
points(coef(mle_model)[1], coef(mle_model)[2], pch = 20, col = "red")

df = reshape2::melt(posterior_contour)
df$beta1 = betaseq[df$Var1]
df$beta2 = betaseq[df$Var2]

neglog_trans = scales::trans_new('-log', 
                                 function(x) -log(-x), function(x) -exp(-x))

ggplot(df, aes(beta1, beta2, z = value)) + 
  geom_raster(aes(fill=value)) + geom_contour(breaks = df_quantiles, colour = "black") +
  geom_point(aes(x = beta[1], y = beta[2]), size = 5, colour = "red") +
  scale_fill_gradientn(trans = neglog_trans, colors = terrain.colors(10), 
                       breaks = c(-100, -500, -3000, -10000), 
                       name = expr) +
  xlab(expression(beta[1])) + ylab(expression(beta[2])) +
  scale_x_continuous(expand = c(0,0)) + scale_y_continuous(expand = c(0,0)) +
  theme_bw()


                       
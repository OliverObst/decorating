# mso graph
t <- seq(1,300,0.1)
alpha <- c(0.2,0.311,0.42,0.51,0.63,0.74,0.85,0.97)

mysin <- function(x,a) {
  sum(sin(a*x))
}

s <- lapply(t, mysin, a = alpha)

plot(t, s, type='l', col='blue', ylab='f(t)')
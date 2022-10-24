# parasin
t <- seq(0,1,0.01)
y1 <- 4*t*(1-t)
y2 <- sin(pi*t)

setEPS()
postscript(file = "parasin2.eps", width = 8, height = 6)
plot(t,y1, type='l', lwd=2, col='blue', ylab='f(t)',cex.axis=1.6, cex.lab=1.5)
lines(t,y2, lwd=3, lty=2, col='red')
dev.off()
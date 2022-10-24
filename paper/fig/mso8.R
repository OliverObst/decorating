library(R.matlab)
mydata <- readMat('../../../examples/mso8/mso8.mat')

setEPS()
postscript(file = "mso8.eps", width = 8, height = 7)
plot(mydata$N,mydata$perc[1,], type='b', xlab=expression(paste('number of reservoir neurons N'^'res')), ylab='success rate', col='red', lty=2, lwd=2, cex.lab=1.5, cex.axis=1.6)
points(mydata$N,mydata$perc[2,], type='b', col='blue', lty=1, lwd=2)
legend("bottomleft", legend=c("T = 150", "T = 130"), lwd=2, lty=1:2, col=c("blue", "red"), cex=1.3)
dev.off()


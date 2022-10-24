library(R.matlab)

mydata <- readMat('../../../examples/parabola_sine/success_rate.mat')

idx <- mydata$N >= 20

setEPS()
postscript(file = "accuracy.eps", width = 8, height = 6)
plot(mydata$N[idx], mydata$parabola[idx], type='b', col='blue', xlab= expression(paste('number of reservoir neurons N'^'res')), ylab='success rate',xlim=c(20,100),ylim=c(0,1.0), lwd=2, cex.axis=1.6, cex.lab=1.5)
lines(mydata$N[idx], mydata$sine[idx], type='b', col='red', lty=2, lwd=2)
legend("bottomright", legend=c("sinusoid", "parabola"), lwd=2, lty=2:1, col=c("red", "blue"), cex=1.3)

dev.off()

library(R.matlab)

mydata <- readMat('../../../examples/ellipse/asymptot.mat')
time=1:1001
error=mydata$Error
setEPS()
postscript(file = "asymptot0.eps", width = 9, height = 5.56)
plot(time,error[1,], type='l', xlab='time t', ylab='distance', col='blue', lty=1, lwd=3, cex.lab=1.5, cex.axis=1.6)
points(time,error[2,], type='l', col='red', lty=2, lwd=4)
points(time, error[3,], type='l', col='black', lty=4, lwd=4)
legend("topright", legend=c("N =   100", "N =   500", "N = 1000"), lwd=4, lty=c(1,2,4), col=c("blue", "red", "black"), cex=1.3)
dev.off()

setEPS()
postscript(file = "asymptot1.eps", width = 9, height = 5.56)
plot(time,error[1,], type='l', xlab='time t', ylab='distance', col='blue', lty=1, lwd=2, cex.lab=1.5, cex.axis=1.6)
points(time,error[2,], type='l', col='red', lty=1, lwd=2)
points(time, error[3,], type='l', col='black', lty=1, lwd=2)
legend("topright", legend=c("N =   100", "N =   500", "N = 1000"), lwd=2, lty=1, col=c("blue", "red", "black"), cex=1.3)
dev.off()

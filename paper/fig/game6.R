#Data <- read.csv('../../../examples/robocup/game6data.csv', header=F)
#Out1 <- read.csv('../../../examples/robocup/game6out1.csv', header=F)
#Out2 <- read.csv('../../../examples/robocup/game6out2.csv', header=F)
library(R.matlab)

mydata <- readMat('game6.mat')

x1 <- mydata$In[1,] # as.numeric(Data[1,])
y1 <- mydata$In[2,] # as.numeric(Data[2,])
x2 <- mydata$Out1[1,]
y2 <- mydata$Out1[2,]
x3 <- mydata$Out2[1,]
y3 <- mydata$Out2[2,]

setEPS()
postscript(file = "game6.eps", width = 12, height = 8)
plot(x2, y2, type='l', lwd=3, lty=1, col='blue', xlab='x position [m]', ylab='y position [m]', xlim=c(-55,55), ylim=c(-45,40), cex.axis=1.6, cex.lab=1.5, panel.first=grid())
lines(x3,y3, lwd=3, lty=2, col='red')
lines(x1,y1, lwd=5, lty=3, col='black')
legend("bottomright", legend=c("original ball trajectory", "LRNN prediction", "reduced LRNN prediction"), lwd=4, lty=c(3,1:2), col=c("black", "blue", "red"), cex=1.4)
dev.off()

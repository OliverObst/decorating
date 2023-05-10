rm(list=ls(all=TRUE))
library ('ggplot2')
library('forecast')
library('tseries')
library('zoo')
#library('hydroGOF')  #for nrmse
#library('TSA') 

#read in data (insert stock name)
  sdata=read.csv('../data/FRE.DE.csv',header=TRUE,sep=",",stringsAsFactors=FALSE)
  sdata$Date = as.Date(sdata$Date)                  #set Date in according format 'Date'
  
  sdata=sdata[1:nrow(sdata),]
  
#separate data into training and test data
  trainlen = 250;
  testlen = 50;  

  #training data
  training_data=tail(head(sdata,-testlen),trainlen)	#create new data frame and fill with training data
  ts_data = ts(training_data[,6])			#declare of adjusted closing price as a time series
  training_data$timeseries = ts(ts_data)		#add time series to training_data 

  #testing data 
  testing_data=tail(sdata,testlen)	#create new data frame and fill with testing data
  ts_data = ts(testing_data[,6])	#set adjusted closing price as time series
  testing_data$timeseries = ts(ts_data)	#add time series to testing_data 
  
  
#build ARIMA/SARIMA model
  
  arima_model=auto.arima(training_data$timeseries, trace=TRUE, approximation=FALSE) #develop arima_model according to timeseries

  tsdisplay(residuals(arima_model),lag.max=15)                                  #check residuals
  training_data$fit=arima_model$fitted                                          #save estimated training data

#predict testing data
  predicted=forecast(arima_model,h=nrow(testing_data))                         #predict testing data
  predicted_tmp=as.numeric(predicted$mean)                                     
  testing_data$Prediction=predicted_tmp

  ggplot()+ 
    labs(title=(expression(atop("ARIMA (2,1,2)",paste(phi[1],'=0.9764, ',phi[2],'=-0.9366, ',theta[1],'=-0.9535, ', theta[2],'=0.8747, ','c=0.1741')))), x="Zeit in t", y="Schlusspreis in ???", col="") +
    theme(plot.title=element_text(hjust=0.5)) +
    geom_line(data=training_data,  aes(x=Date, y = Close, col = "Schlusspreis \n(Trainingsdaten)")) + 
    geom_line(data=training_data,  aes(x=Date, y = fit, col = "Geschaetzter \nSchlusspreis \n(Trainingsdaten)\n"))+  
    geom_line(data=testing_data, aes(x=Date, y = Close, col = "Schlusspreis \n(Testdaten)\n")) +
    geom_line(data=testing_data, aes(x=Date, y = Prediction, col = "Prognostizierter \nSchlusspreis \n(Testdaten)\n")) +
    scale_x_date(date_labels = "%b %y", date_breaks = "6 months")+
    scale_color_manual(values = c("#8B0000","#FA8072","#6495ED","#00008B"))
  
  
#calculate RMSE for training and testing data
#RMSE_train=rmse(training_data$timeseries,training_data$fit)
#  RMSE_train
 
rmse <- function(S,O) {
  len <- min(length(S), length(O))
  diff2 <- (S[1:len] - O[1:len])^2
  rmse <- sqrt(sum(diff2) / len)
}

RMSE_test=rmse(testing_data$timeseries,testing_data$Prediction)
RMSE_test/mean(training_data$timeseries)

write.csv(testing_data$Prediction, "forecast_testing.csv", row.names = FALSE)

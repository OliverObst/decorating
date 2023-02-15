  rm(list=ls(all=TRUE))
  library ('ggplot2')
  library('forecast')
  library('tseries')
  library('zoo')
# library('hydroGOF')  #for nrmse
  
#read in data
  sdata=read.csv('../data/ADS.DE.csv',header=TRUE,sep=",",stringsAsFactors=FALSE)
  sdata$Date = as.Date(sdata$Date)                  #set Date in according format 'Date'
  sdata=sdata[263:nrow(sdata),]
  
#seperate data into training- and testdata  
  training_size = ceiling(0.80 * nrow(sdata))           #set limit for training data
  
  #training data set
  training_data=sdata[1:training_size,]               #create new data frame and fill with training data
  ts_data = ts(training_data[,6])                     #declare of adjusted closing price as a time series
  training_data$timeseries = ts(ts_data)              #add time series to training_data 
  
  #testing data set
  testing_data=sdata[(training_size+1):nrow(sdata),]  #create new data frame and fill with testing data
  ts_data = ts(testing_data[,6])                      #set adjusted closing price as time series
  testing_data$timeseries = ts(ts_data)               #add time series to training_data 

#Seasonal ARIMA model
  K=127
  training_data$timeseries=ts(training_data$timeseries,frequency=254)        
 # sarima_model=auto.arima(ts(training_data$timeseries,frequency=254), trace=TRUE,seasonal=TRUE) #develop sarima_model according to timeseries
  sarima_model=auto.arima(ts(training_data$timeseries),xreg=fourier(training_data$timeseries,K))
  tsdisplay(residuals(sarima_model),lag.max=15)                    #analyse residuals
  training_data$fit=sarima_model$fitted
  training_data$fit=ts(training_data$fit,frequency=254)
#predict testing data
 # predicted=forecast(sarima_model,h=nrow(testing_data))
  predicted=forecast(sarima_model, xreg=fourier(training_data$timeseries,K, h=nrow(testing_data)))
  predicted_tmp=as.numeric(predicted$mean)
  testing_data$Prediction=predicted_tmp
  

  ggplot()+  
    labs(title=(expression(atop("ARIMA (0,1,0)",paste('')))), x="Date", y="stock price in ???", col="") +
    theme(plot.title=element_text(hjust=0.5)) +
    geom_line(data=training_data,  aes(x=Date, y = Close, col = "stock price \n(training data)")) + 
    geom_line(data=training_data,  aes(x=Date, y = fit, col = "estimated \nstock price \n(training data)\n"))+  
    geom_line(data=testing_data, aes(x=Date, y = Close, col = "stock price \n(testing data)\n")) +
    geom_line(data=testing_data, aes(x=Date, y = Prediction, col = "predicted \nstock price \n(testing data)\n")) +
    scale_x_date(date_labels = "%b %y", date_breaks = "6 months")+
    scale_color_manual(values = c("#8B0000","#FA8072","#6495ED","#00008B"))
  
  
#calculate RMSE for training and testing data
#  RMSE_train=rmse(training_data$timeseries,training_data$fit)
#  RMSE_train

rmse <- function(S,O) {
  len <- min(length(S), length(O))
  diff2 <- (S[1:len] - O[1:len])^2
  rmse <- sqrt(sum(diff2) / len)
}

RMSE_test=rmse(testing_data$timeseries,testing_data$Prediction)
RMSE_test


rm(list=ls(all=TRUE))
library ('ggplot2')
library('forecast')
library('tseries')
library('zoo')
#library('hydroGOF')  #for nrmse
#library('TSA') 

total_time <- 0

for (i in 1:20) {
  file_name <- sprintf("../data/signal%02d.csv", i) # Generates file names from signal01.csv to signal20.csv
  results <-  sprintf("forecast_testing_s%02d.csv", i)

  #read in data (mso8 examples)
  sdata=read.csv(file=file_name,header=FALSE,sep=",",stringsAsFactors=FALSE)  
  sdata=t(sdata)
  
  start_time <- Sys.time()
  
  #separate data into training and test data
  trainlen = 250
  testlen = 50 

  #training data
  training_data=head(sdata,trainlen)			#create new data frame and fill with training data
  ts_data = ts(training_data[,1])
  training_data$timeseries = ts(ts_data)		#add time series to training_data 

  #testing data 
  testing_data=head(tail(sdata,-trainlen),testlen)	#create new data frame and fill with testing data
  ts_data = ts(testing_data[,1])
  testing_data$timeseries = ts(ts_data)			#add time series to testing_data 
  
  #Seasonal ARIMA model
  K=100
  N=250
  training_data$timeseries=ts(training_data$timeseries,frequency=N)
  sarima_model=auto.arima(ts(training_data$timeseries,frequency=N),trace=TRUE,seasonal=TRUE) #develop sarima_model according to timeseries
  # sarima_model=auto.arima(ts(training_data$timeseries),xreg=fourier(training_data$timeseries,K))
  tsdisplay(residuals(sarima_model),lag.max=15)                    #analyse residuals
  training_data$fit=sarima_model$fitted
  training_data$fit=ts(training_data$fit,frequency=N)
  #predict testing data
  predicted=forecast(sarima_model)
  # predicted=forecast(sarima_model, xreg=fourier(training_data$timeseries,K, h=nrow(testing_data)))
  predicted_tmp=as.numeric(predicted$mean)
  testing_data$Prediction=predicted_tmp
  
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

  end_time <- Sys.time()
  total_time <-total_time + (end_time - start_time)

  write.csv(testing_data$Prediction, results, row.names = FALSE)
}

average_time <- total_time / 20
print(average_time)

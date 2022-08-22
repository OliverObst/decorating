function [traindat,testdat,names] = read_in_data(trainlen=600,testlen=250)
% read in all stock data from DAX 40 courses (persisting in 2018-2021) 

% names = stock names
% trainlen,testlen : number of trading days for training and testing 
% traindat,testdat : training and testing data sequences 

% definitions
cd('../data/'); % data directory
stocks = dir('*.csv'); % read all file names
n = numel(stocks); % number of stocks

% initialisation
names = cell(1,n);
traindat = zeros(n,trainlen);
testdat = zeros(n,testlen);

% read in data
for k=1:n
  name = nthargout(2,@fileparts,stocks(k).name);
  names(k) = name;
  alldata = csvread(strcat(name,'.csv'),[1,5,1012,5])'; % adjusted closing prices
  testdat(k,:) = alldata(end-testlen+1:end); % help structat end of input 
  traindat(k,:) = alldata(end-testlen-trainlen+1:end-testlen);
endfor

cd('../lrnn'); % go back
endfunction

function [traindat,testdat,names] = read_in_data(trainlen,testlen)
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
  alldat = csvread(strcat(name,'.csv'),[1,5,1012,5])'; % adjusted closing prices
  testdat(k,:) = alldat(end-testlen+1:end); % data from end of input 
  traindat(k,:) = alldat(end-testlen-trainlen+1:end-testlen); % and just before
endfor

cd('../lrnn'); % go back
endfunction

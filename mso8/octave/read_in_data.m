function [traindat,testdat,alpha] = read_in_data(trainlen=250,testlen=50)
% read in all generated mso data from csv files

% names = file names
% trainlen,testlen : number of steps for training and testing 
% traindat,testdat : training and testing data sequences 
%

% definitions
cd('../data/'); % data directory
n = 20; % number of files

% initialisation
traindat = zeros(n,trainlen);
testdat = zeros(n,testlen);
alpha = zeros(n,8);

% read in data
for k=1:n
  alldat = csvread(strcat('signal',num2str(k,'%.2d'),'.csv'));
  traindat(k,:) = alldat(1:trainlen); % data from start 
  testdat(k,:) = alldat(trainlen+(1:testlen));
  alpha(k,:) = csvread(strcat('alpha',num2str(k,'%.2d'),'.csv'));
endfor

cd('../octave/'); % go back
endfunction

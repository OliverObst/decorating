% evaluation script for MSO example

addpath('../prediction');

reps = 100;
signal = mso(1000,8);

n = [130 150 200];
N = 70:10:200;

data = zeros(length(n),length(N),reps);

for i=1:length(n)
  for j=1:length(N)
    for k=1:reps
      [Out,Err,A,J,Y,W,X] = predict(signal(1:n(i)),0,N(j),theta=0.5,delta=0);
      data(i,j,k) = length(J); % number of reduced dimensions
    endfor
  endfor
endfor

perc = mean(data==16,3);

% save data
save('mso8.mat','-v7','n','N','data','perc');

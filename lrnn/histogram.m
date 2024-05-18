function [stats,distr] = histogram(In,N=4,theta=0.5,n=1000)
% continue integer series with reservoir of size N
% for n times and plot histogram and return mode
% strictly speaking, a bar diagram is drawn

y = zeros(1,n);
for (k=1:n)
  Out = predict(In,1,N,theta);
  y(k)= round(Out(1,end)); % predicted next element of series
endfor;

x = unique(y); % different values
f = histc(y,x);
distr = [x;f];
bar(x,f); % draw diagram
stats = [mode(y) mean(y) median(y)]; % most frequent value

endfunction

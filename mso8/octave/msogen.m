%multiple superimposed oscillators

function [signal,alpha] = msogen(n=1000,m=8)

time = 1:n;
alpha = rand(1,m);
signal = zeros(1,n);
for k = 1:m 
    signal += sin(alpha(k)*time);
endfor

%plot(signal)   
csvwrite('signal.csv',signal);
csvwrite('alpha.csv',alpha);

endfunction


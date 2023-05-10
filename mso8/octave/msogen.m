%multiple superimposed oscillators

function [signal,alpha] = msogen(n,m)

  if nargin < 1
    n = 1000;
  end
  if nargin < 2
    m = 8;
  end
  
  time = 1:n;
  if m < 1
    m = 8
    alpha = [0.2,0.311,0.42,0.51,0.63,0.74,0.85,0.97];
  else
    alpha = rand(1,m);
  endif

  signal = zeros(1,n);
  for k = 1:m 
    signal += sin(alpha(k)*time);
  endfor

%plot(signal)   
  csvwrite('signal.csv',signal);
  csvwrite('alpha.csv',alpha);

endfunction

%multiple superimposed oscillators benchmark
%cf. Koryakin, Lohmann, Butz (2012)

function signal = mso(n=1000,m=8,alpha=[0.2 0.311 0.42 0.51 0.63 0.74 0.85 0.97])

time = 1:n;
signal = zeros(1,n);
for k = 1:m 
    signal += sin(alpha(k)*time);
endfor
%plot(signal)   
endfunction


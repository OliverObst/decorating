function f = fibonacci(n)
% compute first n fibonacci numbers

f = ones(1,n);
for (k=3:n)
  f(k) = f(k-1)+f(k-2);
endfor

endfunction


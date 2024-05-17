% predict sequences of natural numbers by recurrent neural networks
% only output prediction function is learned
% rest is random as in echo state networks
%
% (c) Frieder Stolzenburg


% sequences to learn

In = [1 1 2 3 5] % motivating example

a = 2.^(0:5) % powers of 2

b = (0:9) % linear arithmetic series

c = (0:9).^2 % square numbers

d = repmat(1:3,1,3) % oscillation

e = primes(10) % prime numbers up to given limit

f = fibonacci(5) % fibonacci series

g = 2.^(2.^(0:3)) % double exponential

%predict(In,more=1,N=3);
%histogram(In);
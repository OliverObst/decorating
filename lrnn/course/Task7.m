% a = 2.^(0:5) % powers of 2
Wa = [2];
xa = 1;

% b = (0:9) % linear arithmetic series
Wb = [1 1;
      0 1];
xb = [0 1]';

% c = (0:9).^2 % square numbers
Wc = [1 2 1;
      0 1 1;
      0 0 1];
xc = [0 0 1]';

% d = repmat(1:3,1,3) % oscillation
Wd = [0 1 0;
      0 0 1;
      1 0 0];
xd = [1 2 3]';

% e = primes(10) % prime numbers up to given limit
% unknown

% f = fibonacci(5) % fibonacci series
Wf = [0 1;
      1 1];
xf = [0 1]';

% g = 2.^(2.^(0:3)) % double exponential
% not possible with LRNNs

% compute(W,x) ...
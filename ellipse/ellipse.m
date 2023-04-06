function [Error,W,X,Y,lambda,A,J] = ellipse(N=100,n=1000,m=1)
% long-term behavior of random reservoirs
% (c) Frieder Stolzenburg

% N : random reservoir size
% n : number of computation steps
% m : number of trials

% W : transition matrix
% X = W^t*s: original sequence
% Y = A*J^t*y: approximated sequence
% Error : error of Y wrt. X

Error = zeros(1,n+1);

for (k=1:m)
  [W,lambda] = reservoir(N);
  s = randn(N,1); % random start vector ...
  s /= norm(s); % ... with unit norm
  X = compute(W,s,n);

  d = 1+iscomplex(lambda); % asymptotic matrix size
  J = block(lambda,d); % real Jordan block
  y = start(d); % standard start vector
  Y = compute(J,y,n);
  A = X/Y; % matrix mapping Y to X
  Y = A*Y;

  Error += norm(X-Y,2,'columns'); % add actual error
endfor

Error /= m; % mean error
plot(Error);

endfunction

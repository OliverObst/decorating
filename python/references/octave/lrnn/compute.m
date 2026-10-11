function X = compute(W,x,m=10,S=x)
% compute X = W^t*x for t = 0,1,...
% input receiving mode for given input S
% output generating mode for m more steps

% W : recursive function definition (square matrix)
% x : initial column vector (start vector)

d = rows(S);
n = columns(S)-1;
N = rows(x);
X = zeros(N,n+1+m);
X(:,1) = x;

% input receiving mode
X(1:d,1:n+1) = S;
if (n>0)
  I = (d+1):N;
  M = W(I,:);
  for (k=2:n+1)
    X(I,k) = M*X(:,k-1);
  endfor
endif

% output generating mode
if (m>0)
  x = X(:,n+1);
  for (k=n+1+(1:m))
    x = W*x;
    X(:,k) = x;
  endfor
endif

endfunction

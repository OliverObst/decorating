W = [0 1;
      1 1];
x = [0;1];

[V,D] = eigs(W);
%V =
%   0.52573  -0.85065
%   0.85065   0.52573
%
%D =
%  1.61803         0
%  0  -0.61803

% golden ratio
phi = (1+sqrt(5))/2;

% Binet's formula
A = [1/sqrt(5) -1/sqrt(5)];
J = diag([phi 1-phi]);
y = [1 1]';

% whole network
W = [0 1/sqrt(5) -1/sqrt(5);
     0 phi 0;
     0 0 1-phi];
x = [0 phi 1-phi]';
% start with 0 because of delay
    
Fibonacci = [0 fibonacci(29)]
do [Out,Err,A,J,Y,W,X] = predict(Fibonacci,M=10,N=-1,RMSE=0.001);
until (columns(A)==2); full(J)

% evaluation
c=0;
n=100;
Res = zeros(n,2);
for k=1:n;
  [Out,Err,A,J,Y,W,X] = predict(Fibonacci,M=10,N=-1,RMSE=0.001);
  Res(k,:) = [Err,columns(A)];
  if (columns(A)==2)
    c++;
  endif
endfor
[Sort,Index] = sort(Res(:,1));
Res = Res(Index,:)
c

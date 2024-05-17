N = 10;
#W = randn(N);
W = reservoir(N);
x = randn(N,1);

% (a) real eigenvalue +1 or -1 or complex with norm 1
[V,D] = eigs(W,1,'lm')

% (b) behavior converges to oscillation
X = compute(W,x,3000);
#plot(X(1,:)); % plot first dimension

% (c) focus on end section
X = X(:,end-99:end);
plot(X(1,:)); % plot first dimension
%[U,D,V] = svd(X,0);
d = 2;
[Decode,Scale,Error] = princomp(X,d)

% cf. title image

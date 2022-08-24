function [Out,Err,A,J,Y,W,X,Lambda,Multi] = predict(S,M=0,N=-1,theta=-Inf,delta=0,Range=0)
% predict next values of a given input sequence

% S : input sequence (time series)
% M : prediction for M more steps
% N : reservoir size (number of neurons) (-1 = same number of equations and unknowns)
% theta : threshold for error
%   repeat network generation until RMSE < |theta|
%   network size reduction if > 0
% delta : minimal distance between eigenvalues (0 = no clustering)
% Range : indices of relevant input/output components (0 = take all components)

% Out : output sequence (predicted)
% Err : root-mean-square error of predicted output
% A : output matrix (for reduced number of dimensions)
% J : Jordan matrix (for dynamics of reduced version)
% Y : network dynamics state sequence
% W : overall original learned transition matrix [Wout; Win Wres]
%   Wout : learned output weights [Wout]
%   Wir : input and reservoir weight matrix [Win Wres]
% X : original input and reservoir state sequence [S; R]
% Lambda : eigenvalues of J (without eigenvalues with negative imaginery part)
% Multi : their multiplicities (counting complex conjugated eigenvalues twice)
# Omega : angular frequencies for time step tau=1
# Amplitude : corresponding amplitudes for each input/output component
# Phase : corresponding phase shifts for each input/output component


% INITIALISATION

n = columns(S)-1; % sequence length
d = rows(S); % number of inputs, often = 1
if (N<0)
  N = n-d;
endif


% LEARN OUTPUT WEIGHTS

do % repeated reservoir generation

% network initialization (randomly)
W = zeros(d+N);
Index = d+(1:N);
W(Index,1:d) = randn(N,d)/sqrt(d); % balanced input weights [Win]
X = zeros(d+N,n+1); % input and reservoir state sequence
W(Index,Index) = reservoir(N); % Wres
X(Index,1) = start(N);

% drive given input through reservoir (input receiving mode)
X(:,1:n+1) = compute(W,X(:,1),0,S);

% learn output weights
Yout = S(:,2:n+1); % predicted sequence
warning("off","Octave:singular-matrix");
W(1:d,:) = Yout/X(:,1:n); % output weights
warning("on","Octave:singular-matrix");


% REAL JORDAN DECOMPOSITION

% preparation
if (Range==0)
  Range = 1:d;
endif
[Lambda,Multi] = cluster(eig(W),delta); % cluster eigenvalues
Index = find(imag(Lambda)>=0); % only complex numbers with non-negative imaginery part
Lambda = Lambda(Index);
Multi = Multi(Index);
Multi(find(imag(Lambda)>0)) *= 2; % count them twice

% construct real Jordan matrix and determine mapping matrix
[J,N,K] = jormat(Lambda,Multi);
Y = compute(J,start(N),n); % output generating mode
warning("off","Octave:singular-matrix");
A = (X/Y)(Range,:);
warning("on","Octave:singular-matrix");

% initialisation
In = X(Range,:);
Out = A*Y; % predicted sequence
Err = rmse(In,Out); % original error

until (Err < abs(theta))


% NETWORK SIZE REDUCTION

if (theta > 0)

#% compute error for each component
#Error = zeros(K,1);
#pos = 0; % position in Jordan matrix
#for (k=1:K)
#  m = Multi(k);
#  Index = [1:pos pos+m+1:N];
#  JJ = J(Index,Index);
#  YY = compute(JJ,start(N-m),n);
#  warning("off","Octave:singular-matrix");
#  AA = (X/YY)(Range,:);
#  #AA = A(Range,Index); % faster but worse
#  warning("on","Octave:singular-matrix");
#  Outx = AA*YY;
#  Error(k) = rmse(In,Outx);
#  pos += m;
#endfor

#% re-order Jordan components
#[Error,Ord] = sort(Error,"descend");
#Lambda = Lambda(Ord);
#Multi = Multi(Ord);
#[J,N,K] = jormat(Lambda,Multi);
Cumul = cumsum(Multi);

% omit components as long as error remains small enough
k1 = 1;
k2 = K;
while (k1<k2)
  #k = idivide(k1+k2,2); % binary search
  k = floor((k1+k2)/2); % binary search
  NN = Cumul(k);
  JJ = J(1:NN,1:NN);
  YY = compute(JJ,start(NN),n);
  warning("off","Octave:singular-matrix");
  AA = (X/YY)(Range,:);
  warning("on","Octave:singular-matrix");
  Outx = AA*YY;
  Errx = rmse(In,Outx);
  if (Errx<theta)
    Out = Outx;
    Err = Errx;
    A = AA;
    J = JJ;
    Y = YY;
    N = NN;
    k2 = k;
  else
    k1 = k+1;
  endif
endwhile

% remaining eigenvalues and their multiplicities
K = k2;
Lambda = Lambda(1:K);
Multi = Multi(1:K);
endif


% PREDICTION

if (M>0)
  Out = [Out(:,1:end-1) A*compute(J,Y(:,end),M)];
endif


# FREQUENCY ANALYSIS

#K = length(Lambda); % number of real Jordan blocks
#L = rows(A); % number of input/output components

#Omega = angle(Lambda); % angular frequencies for time step tau=1
#%Frequency = Omega/(2*pi*tau); % real frequencies in Hz
#Amplitude = zeros(K,L);
#Phase = zeros(K,L);

#pos = 0; % column position in matrix A
#for (k=1:K)
#  m = Multi(k);
#  Index = pos+(1:m);
#  v = A(:,Index); % matrix segment
#  w = Y(Index,1); % vector segment
#  a = norm(w)*norm(v,2,"rows")'; % amplitude
#  Amplitude(k,:) = a;
#  Phase(k,:) = acos((v*w)'./a);
#  pos += m;
#endfor

endfunction

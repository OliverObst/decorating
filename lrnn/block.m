function J = block(lambda,m)
% build a real Jordan block J for eigenvalue lambda of multiplicity m
% possibly including the complex conjugated eigenvalues

typ = 1+iscomplex(lambda); % type of eigenvalue: 1. real; 2. complex
m /= typ; % proper complex numbers count twice (complex conjugated pair)

J = real(lambda)*eye(typ*m); % main diagonal

if (typ==2) % eigenvalue is complex
  D = diag(repmat([imag(lambda) 0],1,m)(1:end-1));
  J(1:end-1,2:end) += D; % upper diagonal
  J(2:end,1:end-1) -= D; % lower diagonal
endif

if (m>1) % multiplicity > 1
  J(1:end-typ,1+typ:end) += eye(typ*(m-1));
endif

endfunction

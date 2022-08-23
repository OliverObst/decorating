function [J,N,K] = jormat(Lambda,Multi)
% create real Jordan matrix J of size NxN consisting of K jordan blocks
% from list of eigenvalues Lambda with corresponding multiplicities Multi

J = sparse([]); % Jordan matrix_type
N = 0; % matrix size = number of neurons
K = length(Lambda); % number of real Jordan blocks

for (k=1:K)
  m = Multi(k); % block size
  J = blkdiag(J,block(Lambda(k),m));
  N += m;
endfor

endfunction
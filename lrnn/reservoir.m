function [W,lambda] = reservoir(N)
% create reservoir of size N with weight matrix W and unit spectral radius

% N : reservoir size
% W : reservoir transition matrix
% lambda : eigenvalue with absolute value 1 (maximal)

if (N>1)
  while (true)
    try
      W = randn(N); % Gaussian noise square matrix
      lambda = max(eig(W)); % eigenvalue with maximal absolute value
#     lambda = eigs(W,1,'lm'); % does not do the job accurately
      W /= abs(lambda); % set spectral radius to 1
      lambda /= abs(lambda); % normalization of eigenvalue
      break;
    catch
      ;
    end_try_catch
  endwhile
else
  W = ones(N);
endif

endfunction

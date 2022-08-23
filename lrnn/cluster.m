function [Centr,Multi] = cluster(List,delta=0)
% build clusters of the complex numbers (eigenvalues) in List
% up to distance delta for some element in the cluster
% assuming N points lie equidistantly on a unit circle (delta=-1)

% initialization
N = length(List); % number of elements
if (delta<0)
  delta = pi/N; % sin() omitted
endif
Centr = List; % cluster centroids
Multi = ones(N,1); % cluster size
if (delta==0)
  return; % nothing to do
endif
L = 1:N; % links to cluster centroids

% procedure
for i=1:N
  ii = i; % link to new entry
  for j=1:i-1
    jj = L(j);
    while (Multi(jj) == 0)
      L(jj) = L(jj); % follow links
      jj = L(jj);
    endwhile
    if (abs(List(i)-List(j)) < delta) && (ii != jj) % merge clusters
      Centr(jj) = Multi(jj)*Centr(jj)+Multi(ii)*Centr(ii);
      Multi(jj) += Multi(ii);
      Centr(jj) /= Multi(jj);
      Multi(ii) = 0;
      L(ii) = jj; % link to valid entry
      ii = jj;
    endif
  endfor
endfor

% result
Index = find(Multi);
Centr = Centr(Index);
Multi = Multi(Index);

endfunction

function result = rmse(In,Out,Mode=1,Flag=1)
% root-mean-square error

% In : given sequence
% Out : computed sequence
% Mode : kind of normalisation 
% Flag : treatment of dimensions (rows)

result = (Out-In).^2;

switch (Flag)
  case 0 % per row
  case 1 % (Euclidean) distance
    result = sum(result,1);
  case 2 % one (mean) value
    result = mean(result,1);
endswitch

switch (Mode)
  case 0  % without averaging
    result = sum(result,2);
  case 1 % wrt. number of elements (columns)
    result = mean(result,2);
  case 2 % divided by variance
    result = mean(result,2)./var(In,0,2);
endswitch

result = sqrt(result);
endfunction

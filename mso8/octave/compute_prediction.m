% predict mso data and evaluate

% prerequisites
addpath('../../lrnn/');
trainlen = 200;
evallen = 50;
testlen = 50;
trials = 100;

% read in stock data
n = 20; % number of files
[traindat,restdat,alpha] = read_in_data(trainlen,evallen+testlen);

% parameters
Nres = 70;
theta = 0.5;
result = zeros(n,2);

% prediction
for k=1:n
  k#
  In = traindat(k,:);
  Eval = restdat(k,1:evallen); % validation data;
  Test = restdat(k,evallen+1:end); % test data

  Err = +Inf; % best validation error so far
  for t=1:trials
    [Outx,Erry,Ax,Jx,Yx,Wx,Xx,Lambdax,Multix] = predict(In,evallen+testlen,Nres,theta);
    Errx = rmse(Eval,Outx(trainlen+(1:evallen))); % actual validation error
    if Errx<Err
      Err = Errx;
      Out = Outx;
      A = Ax;
      J = Jx;
      Y = Yx;
      W = Wx;
      X = Xx;
      Lambda = Lambdax;
      Multi = Multix;
    endif 
  endfor

  evalerr = result(k,1) = Err % validation error
  testerr = result(k,2) = rmse(Test,Out(end-testlen+1:end)) % test error
  length(J)
endfor

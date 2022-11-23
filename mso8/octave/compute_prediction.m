% predict mso data and evaluate

% prerequisites
addpath('../../lrnn/');
trainlen = 250;
testlen = 50;
trials = 100;

% read in stock data
n = 20; % number of files
[traindat,testdat,alpha] = read_in_data(trainlen,testlen);

% parameters
Nres = 70;
theta = 0.5;
result = zeros(n,2);

% prediction
for k=1:n
  k#
  In = traindat(k,:);
  Err = +Inf; % best training error so far
  for t=1:trials
    [Outx,Errx,Ax,Jx,Yx,Wx,Xx,Lambdax,Multix] = predict(In,0,Nres,theta);
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

  trainerr = result(k,1) = Err % training error
  testerr = result(k,2) = rmse(testdat(k,:),(A(Range,:)*compute(J,Y(:,end),testlen))(2:end)) % test error
  length(J)
endfor

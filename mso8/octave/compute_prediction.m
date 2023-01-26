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
Nres = -1;
theta = 0.5;
result = zeros(n,5);

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

  trainerr = result(k,1) = rmse(In,Out(1:trainlen)) % training error
  evalerr = result(k,2) = Err % validation error
  testerr = result(k,3) = rmse(Test,Out(end-testlen+1:end)) % test error
  netsize = result(k,4) = length(J) % reduced net size
  baseline = result(k,5) = rmse(Test,mean([In Eval])) % baseline = guess average

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

endfor

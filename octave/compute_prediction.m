% predict stock prices and evaluate

% prerequisites
addpath('../lrnn/');
trainlen = 200;
evallen = 50;
testlen = 50;
trials = 100;

% read in stock data
n = 39; % number of stocks
[traindat,restdat,names] = read_in_data(trainlen,evallen+testlen);

% parameter setting
p = 0.1; % relative error
Nres = 10; % reservoir size
result = zeros(n,5);

% prediction of single stock from single/all stocks
for k=1:n#4=BASF
  k##
  In = traindat(k,:);
  Range = 1; Seq = In; % training sequence - single stock
  #Range = k; Seq = traindat(:,:); % training sequence - all stocks
  Eval = restdat(k,1:evallen); % validation data;
  Test = restdat(k,evallen+1:end); % test data
  #hold off; 
  #plot([In Eval Test],'k'); hold on; % real data sequence

  evalerr = +Inf; % best validation error so far
  for t=1:trials
    [Outx,Err,Ax,Jx,Yx,Wx,Xx] = predict(Seq,evallen+testlen,Nres,p*mean(In),delta=0,Range);
    offset = Outx(trainlen+1)-Eval(1);
    evalerrx = rmse(Eval,Outx(trainlen+1:trainlen+evallen)-offset)/mean(In); % actual validation error
    if evalerrx<evalerr
      evalerr = evalerrx;
      Out = Outx;
      A = Ax;
      J = Jx;
      Y = Yx;
      W = Wx;
      X = Xx;
    endif 
  endfor

  meanin = mean([In Eval]); % mean of input including validation phase
  tra = result(k,1) = rmse(In,Out(1:trainlen))/meanin % relative training error
  Raw = compute(W,X(:,1),evallen+testlen,Seq);
  raw = result(k,2) = rmse(Test,Raw(Range,end-testlen+1:end))/meanin % relative testing error
  #plot(Raw(Range,:),'b'); % without network size reduction
  #X = [X(:,1:trainlen-1) compute(W,X(:,trainlen),0,[In(end) Eval])];
  #Y = [Y(:,1:trainlen-1) compute(J,Y(:,trainlen),evallen+testlen)]; % output generating mode
  #A = X/Y(:,1:trainlen+evallen);
  #Dim = A(Range,:)*Y;
  Dim = Out;
  offset = Dim(end-testlen+1)-Test(1);
  dim = result(k,3) = rmse(Test,Dim(end-testlen+1:end)-offset)/meanin % relative testing error
  #plot(Dim,'r'); % with network size reduction
  num = result(k,4) = columns(A) % size of reduced network
  #eva = result(k,5) = evalerr
  bal = result(k,5) = rmse(Test,Eval(end)*ones(1,testlen))/meanin % relative baseline error
  #legend('real price','predicted price','reduced dimensions');
  #xlabel('day (from 2020-10-27 to 2021-12-30)');
  #ylabel('adjusted ending price in € of BAS.DE (BASF)');
endfor

#tra_avg = mean(result(:,1)) % mean training error
#raw_avg = mean(result(:,2)) % mean testing error on raw prediction
#dim_avg = mean(result(:,3)) % mean testing error with size reduction
#num_avg = mean(result(:,4)) % mean net size
#eva_avg = mean(result(:,5)) % mean evaluation error for best nets
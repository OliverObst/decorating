% predict stock prices and evaluate

% prerequisites
addpath('../../lrnn/');
trainlen = 250;
evallen = 50;
testlen = 50;
trials = 1000;

% read in stock data
n = 37; % number of stocks
[traindat,restdat,names] = read_in_data(trainlen,evallen+testlen);

% parameter setting
p = 0.1; % relative error
Nres = 20; % reservoir size
Next = 0; % extra random dimensions
result = zeros(37,5);

% prediction of single stock from single/all stocks
for k=1:n#4=BASF
  k##
  In = traindat(k,:);
  Eval = restdat(k,1:evallen); % validation data;
  Test = restdat(k,evallen+1:end); % test data
  #hold off; 
  #plot([In Eval Test],'k'); hold on; % real data sequence
  
  evalerr = +Inf; % best validation error so far
  for t=1:trials
    R = randn(Next,trainlen);
    #Range = k; Seq = [traindat(:,:); R]; % training sequence - all stocks
    Range = 1; Seq = [In; R]; % training sequence - single stock
    [Outx,Err,Ax,Jx,Yx,Wx,Xx] = predict(Seq,evallen+testlen,Nres,p*mean(In),delta=0,Range);
    evalerrx = rmse(Eval,Outx(trainlen+1:trainlen+evallen))/mean(In); % actual validation error
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
  
  tra = result(k,1) = rmse(In,Out(1:trainlen))/mean(In) % relative training error
  Raw = compute(W,X(:,1),evallen+testlen,Seq);
  raw = result(k,2) = rmse(Test,Raw(Range,end-testlen+1:end))/mean(In) % relative testing error
  #plot(Raw(Range,:),'b'); % without network size reduction
  #X = [X(:,1:end-1) compute(W,X(:,end),evallen)];
  #Y = compute(J,start(columns(J)),trainlen+evallen+testlen-1); % output generating mode
  #A = X/Y(:,1:trainlen+evallen);
  #Dim = A(Range,:)*Y;
  Dim = Out;
  dim = result(k,3) = rmse(Test,Dim(end-testlen+1:end))/mean(In) % relative testing error
  #plot(Dim,'r'); % with network size reduction
  num = result(k,4) = columns(A) % size of reduced network
  eva = result(k,5) = evalerr
  #eva2 = result(k,5) = rmse(Eval,Dim(trainlen+1:trainlen+evallen))/mean(In) % relative validation error 
  #legend('real price','predicted price','reduced dimensions');
  #xlabel('day (from 2020-10-27 to 2021-12-30)');
  #ylabel('adjusted ending price in € of BAS.DE (BASF)');
endfor

raw_avg = mean(result(:,2)) % mean testing error on raw prediction
dim_avg = mean(result(:,3)) % mean testing error with size reduction
num_avg = mean(result(:,4)) % mean evaluation error for best nets
eva_avg = mean(result(:,5)) % mean evaluation error for best nets
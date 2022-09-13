% predict stock prices and evaluate

% prerequisites
addpath('../../lrnn/');
evallen = 50;
trainlen = 250;
testlen = 50;
trials = 100##0;

% read in stock data
n = 37; % number of stocks
[learndat,testdat,names] = read_in_data(evallen+trainlen,testlen);

% parameter setting
p = 0.05; % relative error
Nres = 10; % reservoir size
result = zeros(37,5);

% prediction of single stock from single/all stocks
for k=1##1:n#4=BASF
  k##
  Eval = learndat(k,1:evallen); % validation data;
  In = learndat(k,evallen+1:end); % training data - single stock
  Range = 1; Seq = In; % training sequence - single stock
  #Range = k; In = traindat(:,evallen+1:end) % training sequence - all stocks
  Test = testdat(k,:); % test data
  hold off;
  plot([Eval In Test],'k'); hold on; % real data sequence
  
  evalerr = +Inf; % best validation error so far
  for t=1:trials
    [Outx,Err,Ax,Jx,Yx,Wx,Xx] = predict(Seq,testlen,Nres,p*mean(In),delta=0,Range);
    Backx = (Ax*compute(inv(Jx),Yx(:,1),evallen))(Range,end:-1:2); % reverse computation
    evalerrx = rmse(Eval,Backx)/mean(In); % actual validation error
    if evalerrx<evalerr
      evalerr = evalerrx;
      Out = Outx;
      A = Ax;
      J = Jx;
      Y = Yx;
      W = Wx;
      X = Xx;
      Back = Backx;
    endif 
  endfor
  
  tra = result(k,1) = rmse(In,Out(1:trainlen))/mean(In) % relative training error
  Raw = compute(W,X(:,1),testlen,Seq);
  raw = result(k,2) = rmse(Test,Raw(Range,end-testlen+1:end))/mean(In) % relative testing error
  plot([Eval Raw(Range,:)],'b'); % without network size reduction
  Dim = [Back Out];
  dim = result(k,3) = rmse(Test,Dim(end-testlen+1:end))/mean(In) % relative testing error
  plot(Dim,'r'); % with network size reduction
  num = result(k,4) = columns(A) % size of reduced network
  eva = result(k,5) = evalerr
  #legend('real price','predicted price','reduced dimensions');
  #xlabel('day (from 2020-10-27 to 2021-12-30)');
  #ylabel('adjusted ending price in € of BAS.DE (BASF)');
endfor

raw_avg = mean(result(:,2)) % mean testing error on raw prediction
dim_avg = mean(result(:,3)) % mean testing error with size reduction
num_avg = mean(result(:,4)) % mean evaluation error for best nets
eva_avg = mean(result(:,5)) % mean evaluation error for best nets
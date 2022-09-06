% predict stock prices and evaluate

% prerequisites
addpath('../../lrnn/');
trainlen = 250;
testlen = 50;

% read in stock data
n = 37; % number of stocks
[traindat,testdat,names] = read_in_data(trainlen,testlen);

% parameter setting
p = 0.01; % relative error
Nres = 0; % reservoir size
Next = 200; % extra random dimensions
result = zeros(37,3);

% prediction of single stock from single/all stocks
for k=16##1:n#4=BASF
  k##
  In = traindat(k,:);
  Seq = [traindat(k,:); randn(Next,trainlen)]; % training sequence - sigle stock
  #Seq = [traindat(:,:); randn(Next,trainlen)]; % training sequence - all stocks
  Test = testdat(k,:); % test data
  #hold off; 
  plot([In Test],'k'); hold on; % real data sequence
  [Out,Err,A,J,Y,W,X,Lambda,Multi] = predict(Seq,testlen,Nres,p*mean(In),delta=0,Range=k);
  tra = rmse(In,Out(1:trainlen))/mean(In) % training error
  Raw = compute(W,X(:,1),testlen,Seq)(k,:);
  raw = result(k,1) = rmse(Test,Raw(trainlen+1:end))/mean(In) % relative testing error
  plot(Raw,'b'); % without network size reduction
  Dim = Out;
  dim = result(k,2) = rmse(Test,Dim(trainlen+1:end))/mean(In) % relative testing error
  plot(Dim,'r'); % with network size reduction
  num = result(k,3) = columns(A) % size of reduced network
  #legend('real price','predicted price','reduced dimensions');
  #xlabel('day (from 2020-10-27 to 2021-12-30)');
  #ylabel('adjusted ending price in € of BAS.DE (BASF)');
endfor

org_avg = mean(result(:,1)) % print mean testing error
raw_avg = mean(result(:,2)) % print mean testing error

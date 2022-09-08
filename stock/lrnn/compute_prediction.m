% predict stock prices and evaluate

% prerequisites
addpath('../../lrnn/');
trainlen = 250;
testlen = 50;

% read in stock data
n = 37; % number of stocks
[traindat,testdat,names] = read_in_data(trainlen,testlen);
#mu = mean(traindat,2);
#si = std(traindat,1,2);
#traindat = (traindat-mu)./si;
#testdat = (testdat-mu)./si;

% parameter setting
p = 0.05; % relative error
Nres = 0; % reservoir size
Next = 150; % extra random dimensions
result = zeros(37,5);
R = randn(Next,trainlen);

% prediction of single stock from single/all stocks
for k=1:n#4=BASF
  k##
  In = traindat(k,:);
  Range = k; Seq = [traindat(:,:); R]; % training sequence - all stocks
  #Range = 1; Seq = [In; R]; % training sequence - single stock
  Test = testdat(k,:); % test data
  #hold off; 
  #plot([In Test],'k'); hold on; % real data sequence
  [Out,Err,A,J,Y,W,X,Lambda,Multi] = predict(Seq,testlen,Nres,p*mean(In),delta=0,Range);
  tra = result(k,1) = rmse(In,Out(1:trainlen))/mean(In) % relative training error
  Raw = compute(W,X(:,1),testlen,Seq)(Range,:);
  raw = result(k,2) = rmse(Test,Raw(trainlen+1:end))/mean(In) % relative testing error
  #plot(Raw,'b'); % without network size reduction
  Dim = Out;
  dim = result(k,3) = rmse(Test,Dim(trainlen+1:end))/mean(In) % relative testing error
  #plot(Dim,'r'); % with network size reduction
  num = result(k,4) = columns(A) % size of reduced network
  #legend('real price','predicted price','reduced dimensions');
  #xlabel('day (from 2020-10-27 to 2021-12-30)');
  #ylabel('adjusted ending price in € of BAS.DE (BASF)');
  bal = result(k,5) = rmse(Test,In(end)*ones(1,testlen))/mean(In) % relative baseline error 
endfor

#raw_avg = mean(result(:,2)) % mean testing error on raw prediction
#dim_avg = mean(result(:,3)) % mean testing error with size reduction

#[p Range Nres Next 0 raw_avg dim_avg]
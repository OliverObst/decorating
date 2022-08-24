function [means,stds,theta,rho] = evaluate(name,val,In,nums,step,reps);

data = zeros(reps,4);
means = zeros(nums,4);
stds = zeros(nums,4);

for m=1:nums
  N=step*m;
  for r=1:reps
    tic;
    [Out,Err,A,J,Y,W,X] = predict(In,0,N,theta=0.01,delta=0.03);
    tim = toc; % run-time
    dim = columns(J); % number of reduced dimensions
    scc = dim==val;
    mse = rmse(In,Out); % root-mean-square error
    data(r,:) = [scc dim mse tim]; 
  endfor
  means(m,:) = mean(data,1);
  stds(m,:) = std(data,1);
endfor

stds(:,1) = min(max(means(:,1)),stds(:,1)); % otherwise standard deviations of mse are too big
head = ['success';'dimens';'rmse';'time'];
scale = step*(1:nums)';
for k=1:4
  diagram = figure(k);
  errorbar(scale,means(:,k)',stds(:,k)');
  file = strcat(name,'_',head(k,:),'.png');
# print(diagram,'-dpng','-color',file);
endfor

% polynomial regression for runtimes
x = log(scale);
X = [ones(nums,1) x];
y = log(means(:,3));
theta = (X'*X)\X'*y;
rho = corr(x,y);

% save data
format long;
save(strcat(name,'.out'),'scale','means','stds','theta','rho');

endfunction

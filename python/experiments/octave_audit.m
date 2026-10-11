% Audit the pinned, unmodified Octave routines with shared and native weights.
args = argv();
addpath(args{1});
load(args{2}); % S, Win, Wres, train_length, forecast_length
prefix = args{3}; trials = str2num(args{4});
warning('off', 'all');
observed = S(:,1:train_length); d = rows(observed); r = rows(Win); n = train_length-1;
W = zeros(d+r); W(d+1:end,:) = [Win Wres];
X = compute(W, [observed(:,1); start(r)], 0, observed);
W(1:d,:) = observed(:,2:end)/X(:,1:end-1);
[Lambda,Multi] = cluster(eig(W),0);
idx = find(imag(Lambda)>=0); Lambda = Lambda(idx); Multi = Multi(idx);
Multi(imag(Lambda)>0) *= 2;
[J,N,K] = jormat(Lambda,Multi); Error = zeros(K,1); pos = 0;
for k=1:K
 m = Multi(k); idx = [1:pos pos+m+1:N]; JJ = J(idx,idx);
 YY = compute(JJ,start(N-m),n); AA = observed/YY;
 Error(k) = rmse(observed,AA*YY); pos += m;
endfor
[Error,Ord] = sort(Error,'descend'); Lambda = Lambda(Ord); Multi = Multi(Ord);
% Explicit final-prefix audit agrees with Python; native trials below retain
% the unmodified predict.m boundary and retry behaviour.
for count=1:K
 [J,N,Kx] = jormat(Lambda(1:count),Multi(1:count));
 Y = compute(J,start(N),n); A = observed/Y;
 training_rmse = rmse(observed,A*Y);
 if training_rmse<=.5; break; endif
endfor
Out = A*compute(J,Y(:,end),forecast_length);
future_rmse = rmse(S(:,train_length+1:end),Out(:,2:end));
eigenvalues = eig(W); octave_version = version();
save('-mat7-binary',[prefix '_fixed.mat'],'W','X','eigenvalues','N', ...
     'training_rmse','future_rmse','octave_version');
f = fopen([prefix '_trials.csv'],'w');
fprintf(f,'seed,dimension,training_rmse,future_rmse\n');
for seed=0:trials-1
 randn('seed',seed); % explicit legacy global RNG, independent of Python RNG
 [Out,Err,A,J] = predict(observed,forecast_length,r,.5,0);
 fprintf(f,'%d,%d,%.17g,%.17g\n',seed,rows(J), ...
         rmse(observed,Out(:,1:train_length)), ...
         rmse(S(:,train_length+1:end),Out(:,train_length+1:end)));
 fflush(f);
endfor
fclose(f);

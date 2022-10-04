% A minimalistic sparse Echo State Networks demo with Mackey-Glass (delay 17) data 
% in "plain" Matlab/Octave.
% from https://mantas.info/code/simple_esn/
% (c) 2012-2020 Mantas Lukosevicius
% Distributed under MIT license https://opensource.org/licenses/MIT

# 2022 adapted by Frieder Stolzenburg
addpath('../lrnn/');

% load the data
initLen = 50;
trainLen = 200;
testLen = 50;

n = 39; % number of stocks
[restdat,testdat,names] = read_in_data(initLen+trainLen,testLen);
stock = [restdat testdat];

result = zeros(1,n);

% generate the ESN reservoir
inSize = 1; outSize = 1;
resSize = 10;
a = 0.3; % leaking rate
#rand( 'seed', 42 );

for k=1:n#4=BASF

k##
data = stock(k,:)';

#% plot some of it
#figure(10);
#plot(data);
#title('A sample of data');

Win = (rand(resSize,1+inSize)-0.5) .* 1;
% dense W:
W = rand(resSize,resSize)-0.5;
% sparse W:
% W = sprand(resSize,resSize,0.01);
% W_mask = (W~=0); 
% W(W_mask) = (W(W_mask)-0.5);

% normalizing and setting spectral radius
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* (1.25 / rhoW);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
	u = data(t);
	x = (1-a)*x + a*tanh( Win*[1;u] + W*x );
	if t > initLen
		X(:,t-initLen) = [1;u;x];
	end
end

% train the output by ridge regression
reg = 1e-8;  % regularization coefficient
% direct equations from texts:
%X_T = X'; 
%Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
% using Matlab mldivide solver:
Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))'; 

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize, testLen);
u = data(trainLen+1);
for t = 1:testLen 
	x = (1-a)*x + a*tanh( Win*[1;u] + W*x );
	y = Wout*[1;u;x];
	Y(:,t) = y;
	% generative mode:
	u = y;
	% this would be a predictive mode:
	%u = data(trainLen+t+1);
end

% compute MSE for the first errorLen time steps
#errorLen = testLen;
#mse = sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen;
#disp( ['MSE = ', num2str( mse )] );
meanin = mean(data(initLen+trainLen)); % mean of input including initialization phase
test_rmse = result(k) = rmse(data(end-testlen+1:end),Y(1,1:testLen))/meanin % relative testing error

endfor

#% plot some signals
#figure(1);
#plot( data(trainLen+2:trainLen+testLen+1), 'color', [0,0.75,0] );
#hold on;
#plot( Y', 'b' );
#hold off;
#axis tight;
#title('Target and generated signals y(n) starting at n=0');
#legend('Target signal', 'Free-running predicted signal');
#
#figure(2);
#plot( X' );
#title('Some reservoir activations x(n)');
#
#figure(3);
#bar( Wout' )
#title('Output weights W^{out}');
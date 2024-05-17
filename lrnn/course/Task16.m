% constant bias as remedy
#biased = [99:-1:0;ones(1,100)];

% remembering last but one value
n = 50;
In = repmat(randn(1,n),3,1); % identical rows
In(2,:) = shift(In(2,:),1); % previous element (simple)
In(3,:) = shift(In(3,:),2); % shifted once more (memory)
a = In(1:2,:);
b = In;
In(2,:) = []; c = In;

% cue for recursion formula (data augmentation)
#fibo = [fibonacci(n); 0 fibonacci(n-1)];

% single and double exponential
#t = 0:4; simpl = (2.^t); doubl = [simpl;2.^simpl];
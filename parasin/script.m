% evaluation script
addpath('../prediction');

% experimental setting
step = 5;
nums = 20;
reps = 100;

% parabola and sine
tau = 0.01;
t = 0:tau:1;
In1 = 4*t.*(1-t);
In2 = sin(pi*t);

evaluate('parabola',3,In1,nums,step,reps);
evaluate('sine',2,In2,nums,step,reps);

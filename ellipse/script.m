addpath('../lrnn');

N = [100;500;1000]; % reservoir sizes
Error = zeros(3,1001); % averaged errors

for (k=1:3)
  Error(k,:) = ellipse(N(k),n=1000,m=1000); % 1000 trials
endfor;

save("-v7","asymptot.mat","N","Error");

hold on;
plot(Error(1,:),"color","blue")
plot(Error(2,:),"color","red")
plot(Error(3,:),"color","black")
xlabel("time t");
ylabel("distance");
text(200,0.2,"N =   100","color","blue");
text(200,0.25,"N =   500","color","red");
text(200,0.3,"N = 1000","color","black");
hold off;
print(figure(1),'-dpdf','-color','test.pdf');

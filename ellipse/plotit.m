addpath('../lrnn');

N=100;
W = reservoir(N);
s = start(N);
X = compute(W,s,1000);

diagram=figure(1);
plot(eig(W),'Markersize',20,'k.');
axis([-1,1,-1,1],"equal","on");
#print(figure(1),'-dpdf','-color','spectrum.pdf');

diagram=figure(2);
plot(X(1,:),X(2,:),'Markersize',20,'r.');
axis("equal","off");
#print(figure(2),'-dpdf','-color','ellipse.pdf');

diagram=figure(3);
plot(X(1,1:500),'LineWidth',2,'b-');
axis([0,500,-0.13,0.1],"on");
#print(figure(3),'-dpdf','-color','wave.pdf');

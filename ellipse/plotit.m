addpath('../lrnn');
__mfile_encoding__("utf-8");
clear functions

N=100;
W = reservoir(N);
s = start(N);
X = compute(W,s,1000);

diagram=figure(1);
plot(eig(W),'MarkerSize',15,'k.');
axis([-1,1,-1,1],'equal','on');
axe=get(gcf,'currentaxes');
set(axe,'Fontsize',20,'LineWidth',2);
xlabel('Re(λ)','Fontsize',25);
ylabel('Im(λ)','Fontsize',25);
print(figure(1),'-svgconvert','-dpdf','-color','spectrum.pdf');

diagram=figure(2);
plot(X(1,:),X(2,:),'MarkerSize',15,'r.');
axis('equal','on');
axe=get(gcf,'currentaxes');
set(axe,'Fontsize',20,'LineWidth',2);
xlabel('x','Fontsize',25);
ylabel('y','Fontsize',25);
print(figure(2),'-dpdf','-color','ellipse.pdf');

diagram=figure(3);
plot(X(1,1:500),'LineWidth',2,'b-');
axis([0,500,-0.18,0.18],'on');
axe=get(gcf,'currentaxes');
set(axe,'Fontsize',20,'LineWidth',2);
xlabel('t','Fontsize',25);
ylabel('f(t)','Fontsize',25);
print(figure(3),'-dpdf','-color','wave.pdf');

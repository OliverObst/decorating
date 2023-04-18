function [DIMENS,RMSE1,RMSE2,CHANCE] = evaluate(N=1000,theta=1.0,step=20,I=1:2)
# goalkeeper must be at index I=1:2

% name : file name prefix (string)
% N : number of reservoir neurons
% theta : RMSE threshold for dimension reduction
% I : indexes of interesting input components
% step : step width

#load(strcat(name,'.in'));
load('CYRUS_FRA-UNIted.in'); Org1 = Data(:,1:step:end);
load('FRA-UNIted_HELIOS.in'); Org2 = Data(:,1:step:end);
load('opusSCOM_FRA-UNIted.in'); Org3 = Data(:,1:step:end);
load('FRA-UNIted_opuHam.in'); Org4 = Data(:,1:step:end);
load('FRA-UNIted_Fifty-Storms.in'); Org5 = Data(:,1:step:end);
load('FRA-UNIted_Ri-one.in'); Org6 = Data(:,1:step:end);
load('Persepolis_FRA-UNIted.in'); Org7 = Data(:,1:step:end);
#Org = Data(:,1:step:end);
Org = [Org1 Org2 Org3 Org4 Org5 Org6 Org7];

In = Org(I,:);
len = columns(In);

[Out2,Err,A,J,Y,W,X] = predict(Org,0,N,theta,0,I);

x0=X(:,1);
Out1 = compute(W,x0,len-1)(I,:);

RMSE1 = rmse(In,Out1);
RMSE2 = rmse(In,Out2);

DIMENS = columns(J);
CHANCE = rmse(In,In(randperm(len)));
save('JapanOpen2020.out','-binary','In','Out1','Out2','A','J','Y','W','X','DIMENS','RMSE1','RMSE2','CHANCE');

diagram = figure(1);
hold off;
plot(In(1,:),In(2,:),'Color','black');
hold on;
plot(Out1(1,:),Out1(2,:),'Color','blue');
plot(Out2(1,:),Out2(2,:),'Color','red');
hold off;
print(diagram,'-deps','-color','JapanOpen2020.eps');

endfunction

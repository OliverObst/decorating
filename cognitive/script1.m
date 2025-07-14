S = [12,15,8,11,4,7,0, 3;
     148,84,52,36,28,24,22, 21;
     2,12,21,29,36,42,47, 51;
     2,3,5,9,17,33,65, 129;
     2,5,8,11,14,17,20, 23;
     2,5,9,19,37,75,149, 299;
     25,22,19,16,13,10,7, 4;
     28,33,31,36,34,39,37, 42;
     3,6,12,24,48,96,192, 384;
     3,7,15,31,63,127,255, 511;
     4,11,15,26,41,67,108, 175;
     5,6,7,8,10,11,14, 15;
     54,48,42,36,30,24,18, 12;
     6,8,5,7,4,6,3, 5;
     6,9,18,21,42,45,90, 93;
     7,10,9,12,11,14,13, 16;
     8,10,14,18,26,34,50, 66;
     8,12,10,16,12,20,14, 24;
     8,12,16,20,24,28,32, 36;
     9,20,6,17,3,14,0, 11];
theta = 0.1;
n = 1000;

addpath('../lrnn');

Data = zeros(20,5,4);

for k=1:20
  In = S(k,1:end-1);
  Out = S(k,end);
  [Datum,Distr] = histogram(In,3,-theta,n); % Nres = 3
  Distr = [Distr,[Out;0]];
  Data(k,1,:) = [Datum Distr(2,find(Distr(1,:)==Out,1))];
  
  [Datum,Distr] = histogram(In,4,-theta,n); % Nres = 4
  Distr = [Distr,[Out;0]];
  Data(k,2,:) = [Datum Distr(2,find(Distr(1,:)==Out,1))];
  
  [Datum,Distr] = histogram(In,5,-theta,n); % Nres = 5
  Distr = [Distr,[Out;0]];
  Data(k,3,:) = [Datum Distr(2,find(Distr(1,:)==Out,1))];

  [Datum,Distr] = histogram(In,-1,theta,n); % without clue
  Distr = [Distr,[Out;0]];
  Data(k,4,:) = [Datum Distr(2,find(Distr(1,:)==Out,1))];

  [Datum,Distr] = histogram(clue(In),-1,theta,n); % with clue
  Distr = [Distr,[Out;0]];
  Data(k,5,:) = [Datum Distr(2,find(Distr(1,:)==Out,1))];
endfor

Perc1 = Data(:,:,4)/n;

Out = S(:,end);
Res = Data(:,:,1)==Out;
Perc2 = mean(Res,1);

% save data 
format long;
save('puzzles.out','S','Data','Out','Perc1','Perc2');


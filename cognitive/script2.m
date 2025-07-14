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

addpath('../lrnn');

Data = zeros(20,2);

# find small network size
Data(:,1) = Inf;
for i=1:20
  for j=1:1000
    [Out,Err,A,J,Y,W,X] = predict(S(i,1:end-1),1,-1,theta);
    Data(i,1) = min(Data(i,1),columns(A));
  endfor
endfor

# evaluate prediction
for i=1:20
  for j=1:1000
    do
      [Out,Err,A,J,Y,W,X] = predict(S(i,1:end-1),1,-1,theta);
    until (columns(A)==Data(i,1));
    if all(round(Out)==S(i,:))
        Data(i,2)++;
      endif
  endfor
endfor

# output result
Data
save -binary result.out Data

##for k=20  # examples
##  i=0;      # number of generated reservoirs
##  j=0;      # number of minimal reservoirs
##  m = Inf;  # size of smallest reservoir
##  n = 0;    # number of correct predictions
##  while i<1000 || j<1000
##    [Out,Err,A,J,Y,W,X] = predict(S(k,1:end-1),1,-1,theta);
##    i++;
##    c = columns(A);
##    round(Out)-S(k,:)
##    if c==m
##      j++;
##      if all(round(Out)==S(k,:))
##        n++;
##      endif
##    endif
##    if c<m
##      m = c#;
##      j = 1;
##      n = all(round(Out)==S(k,:));
##    endif
##  endwhile
##  Data(k,:)= [m n];
##endfor


function extract(name);
% reduce data to time step and all (x,y) coordinates

Index = [97,98,105,106,113,114,121,122,129,130,137,138,145,146,153,154,162,163,170,171,178,179,...#right team
9,10,17,18,25,26,33,34,41,42,49,50,57,58,65,66,73,74,81,82,89,90,...#left team
5,6];% ball

Data = csvread(strcat(name,'.csv'));
Data = Data(:,Index)';
Data = repmat([-1;1],23,columns(Data)).*Data;#all games from left

save('-binary',strcat(name,'.in'),'Data');

endfunction

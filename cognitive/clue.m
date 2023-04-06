function Out = clue(In,k=1)

Out = fliplr(hankel(fliplr(In)))(1:k+1,k+1:end);

endfunction
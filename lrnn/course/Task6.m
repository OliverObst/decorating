>> A = [1 2 3; 4 5 6; 7 8 9]
A =

   1   2   3
   4   5   6
   7   8   9

>> A'
ans =

   1   4   7
   2   5   8
   3   6   9

>> A(1:2,2:end)
ans =

   2   3
   5   6

>> B = randn(3,3)
B =

  -0.706924  -0.883785  -0.107676
  -2.187851   1.438591   0.387977
  -0.044543  -0.405569  -0.401513

>> abs(B)
ans =

   0.706924   0.883785   0.107676
   2.187851   1.438591   0.387977
   0.044543   0.405569   0.401513

>> norm(B)
ans = 2.6597
>> help norm
'norm' is a built-in function from the file libinterp/corefcn/data.cc

 -- norm (A)
 -- norm (A, P)
 -- norm (A, P, OPT)
     Compute the p-norm of the matrix A.

     If the second argument is not given, 'p = 2' is used.

     If A is a matrix (or sparse matrix):

     P = '1'
          1-norm, the largest column sum of the absolute values of A.

     P = '2'
          Largest singular value of A.

     P = 'Inf' or "inf"
          Infinity norm, the largest row sum of the absolute values of
          A.

     P = "fro"
          Frobenius norm of A, 'sqrt (sum (diag (A' * A)))'.

     other P, 'P > 1'
          maximum 'norm (A*x, p)' such that 'norm (x, p) == 1'

     If A is a vector or a scalar:

     P = 'Inf' or "inf"
          'max (abs (A))'.

     P = '-Inf'
          'min (abs (A))'.

     P = "fro"
          Frobenius norm of A, 'sqrt (sumsq (abs (A)))'.

     P = 0
          Hamming norm--the number of nonzero elements.

     other P, 'P > 1'
          p-norm of A, '(sum (abs (A) .^ P)) ^ (1/P)'.

     other P 'P < 1'
          the p-pseudonorm defined as above.

     If OPT is the value "rows", treat each row as a vector and compute
     its norm.  The result is returned as a column vector.  Similarly,
     if OPT is "columns" or "cols" then compute the norms of each column
     and return a row vector.

     See also: normest, normest1, vecnorm, cond, svd.

Additional help for built-in functions and operators is
available in the online version of the manual.  Use the command
'doc <topic>' to search the manual index.

Help and information about Octave is also available on the WWW
at https://www.octave.org and via the help@octave.org
mailing list.

>> round(B)
ans =

  -1  -1   0
  -2   1   0
   0   0   0


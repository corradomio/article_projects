from stdlib.iset import *

n = 128
M = 2**n-1

L = ibinlist(M, n)
I = ibinset(L)

print(L)
print("%X" % I)

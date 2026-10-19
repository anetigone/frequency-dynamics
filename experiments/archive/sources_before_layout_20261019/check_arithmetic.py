"""Independent Fraction checks of fixed-point primitives, including signed operands."""
from interval_core import I,S
from fractions import Fraction as F
import random
r=random.Random(31)
for _ in range(10000):
 a,b=sorted(r.randint(-10*S,10*S) for j in range(2));c,d=sorted(r.randint(-10*S,10*S) for j in range(2))
 x,y=I(a,b),I(c,d)
 for z,vs in [(x+y,[F(i+j,S) for i in (a,b) for j in (c,d)]),(x*y,[F(i*j,S*S) for i in (a,b) for j in (c,d)])]:
  assert F(z.lo,S)<=min(vs)<=max(vs)<=F(z.hi,S)
 if c*d>0:
  z=x/y;vs=[F(i,j) for i in (a,b) for j in (c,d)]
  assert F(z.lo,S)<=min(vs)<=max(vs)<=F(z.hi,S)
print('10000 exact Fraction arithmetic checks passed')

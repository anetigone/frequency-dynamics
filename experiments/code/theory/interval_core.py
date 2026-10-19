"""Validated Taylor integration. Python standard library only; no float in proof.
State = w1,w2,B11,B12,B21,B22,g1,g2. Fixed-point interval endpoints.
"""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse,json,time
S=1<<180
class I:
    __slots__=('lo','hi')
    def __init__(self,lo,hi=None): self.lo,self.hi=lo,lo if hi is None else hi
    @staticmethod
    def rat(p,q=1): return I(p*S//q,-((-p*S)//q))
    def __add__(a,b):
        if isinstance(b,int): b=I.rat(b)
        return I(a.lo+b.lo,a.hi+b.hi)
    __radd__=__add__
    def __neg__(a): return I(-a.hi,-a.lo)
    def __sub__(a,b): return a+-b
    def __mul__(a,b):
        if isinstance(b,int): return I(a.lo*b,a.hi*b) if b>=0 else I(a.hi*b,a.lo*b)
        z=[a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi]
        return I(min(z)//S,-((-max(z))//S))
    __rmul__=__mul__
    def __truediv__(a,b):
        if isinstance(b,int): b=I.rat(b)
        assert b.lo*b.hi>0,'denominator contains zero'
        nums=[a.lo*S,a.hi*S]; den=[b.lo,b.hi]
        return I(min(n//d for n in nums for d in den),max(-((-n)//d) for n in nums for d in den))
    def inside(a,b): return b.lo<a.lo and a.hi<b.hi
    def hull(a,b): return I(min(a.lo,b.lo),max(a.hi,b.hi))
    def out(a): return [str(a.lo),str(a.hi)]
    def show(a): return [a.lo/S,a.hi/S] # display only
Z=I(0); ONE=I(S)
class J:
    def __init__(self,a): self.a=a
    def __add__(a,b):
        if isinstance(b,int): b=J([I.rat(b)]+[Z]*(len(a.a)-1))
        return J([x+y for x,y in zip(a.a,b.a)])
    __radd__=__add__
    def __neg__(a): return J([-x for x in a.a])
    def __sub__(a,b): return a+-b
    def __mul__(a,b):
        if isinstance(b,int): return J([x*b for x in a.a])
        return J([sum((a.a[k]*b.a[n-k] for k in range(n+1)),Z) for n in range(len(a.a))])
    __rmul__=__mul__
    def __truediv__(a,b):
        if isinstance(b,int): return J([x/b for x in a.a])
        c=[]
        for n in range(len(a.a)):
            c.append((a.a[n]-sum((b.a[k]*c[n-k] for k in range(1,n+1)),Z))/b.a[0])
        return J(c)
def quantities(x):
    w,v,b,c,d,e,g,j=x
    a=b*w+d*v; z=c*w+e*v
    V=a*a+z*z
    p=b*b+d*d+4*g*(b*w*w+d*v*v)
    q=c*c+e*e+4*j*(c*w*w+e*v*v)
    r=b*c+d*e
    H=a*(p*a+r*z)+z*(r*a+q*z)
    return a,z,V,H

def field(x):
    w,v,b,c,d,e,g,j=x
    a,z,V,H=quantities(x)
    fac=V/(2*H)
    zero=g*0
    return [(b*a+c*z)*fac,(d*a+e*z)*fac,4*g*b*w*a*fac,4*j*c*w*z*fac,4*g*d*v*a*fac,4*j*e*v*z*fac,zero,zero]

def coeff(x,n):
    a=[[v] for v in x]
    for k in range(n):
        f=field([J(v) for v in a])
        for i in range(8): a[i].append(f[i].a[k]/(k+1))
    return a

def risk(x):
    a,z,V,H=quantities(x)
    assert a.lo>0 and z.lo>0
    low=(I(z.lo)*I(z.lo))/(I(a.hi)*I(a.hi)+I(z.lo)*I(z.lo))
    high=(I(z.hi)*I(z.hi))/(I(a.lo)*I(a.lo)+I(z.hi)*I(z.hi))
    return I(low.lo,high.hi)

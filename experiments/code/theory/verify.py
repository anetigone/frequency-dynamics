"""Log-coordinate Euclidean perturbation bound around validated center solution."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
from interval_core import I,J,Z,ONE,S,field,quantities,coeff,risk
import json,time
class D:
    def __init__(self,v,d=None): self.v=v; self.d=[Z]*8 if d is None else d
    def __add__(a,b):
        if isinstance(b,int): b=D(I.rat(b))
        return D(a.v+b.v,[x+y for x,y in zip(a.d,b.d)])
    __radd__=__add__
    def __neg__(a): return D(-a.v,[-x for x in a.d])
    def __sub__(a,b): return a+-b
    def __mul__(a,b):
        if isinstance(b,int): return D(a.v*b,[x*b for x in a.d])
        return D(a.v*b.v,[x*b.v+a.v*y for x,y in zip(a.d,b.d)])
    __rmul__=__mul__
    def __truediv__(a,b):
        if isinstance(b,int): return D(a.v/b,[x/b for x in a.d])
        return D(a.v/b.v,[(x-(a.v/b.v)*y)/b.v for x,y in zip(a.d,b.d)])
def logjac(x):
    y=[D(v,[v if i==j else Z for j in range(8)]) for i,v in enumerate(x)]
    f=field(y)
    return [(f[i]/y[i]).d for i in range(8)]
def pd_bound(A,m):
    # Separate interval uncertainty using the symmetric error's infinity norm.
    sym=[[(A[i][j]+A[j][i])/2 for j in range(8)] for i in range(8)]
    mid=[[I((v.lo+v.hi)//2) for v in row] for row in sym]
    err=max(sum(max(v.hi-c.lo,c.lo-v.lo) for v,c in zip(row,cr)) for row,cr in zip(sym,mid))
    m=m-I(err)
    A=mid
    # Interval LDL^T of m I - sym(A): positive pivots certify all matrices.
    H=[[(m if i==j else Z)-(A[i][j]+A[j][i])/2 for j in range(8)] for i in range(8)]
    L=[[Z]*8 for _ in range(8)]; piv=[]
    for i in range(8):
        d=H[i][i]-sum((L[i][k]*L[i][k]*piv[k] for k in range(i)),Z)
        if d.lo<=0:return False
        piv.append(d);L[i][i]=ONE
        for j in range(i+1,8): L[j][i]=(H[j][i]-sum((L[j][k]*L[i][k]*piv[k] for k in range(i)),Z))/d
    return True

def run(den=100000):
    stepden=500; order=8
    x=[I.rat(1,25),I.rat(9,25),I.rat(8,25),I.rat(4,125),I.rat(4,125),I.rat(8,125),I.rat(1),I.rat(2)]
    perturb=I.rat(den-1,den).hull(I.rat(den+1,den)); initial=[v*perturb for v in x]
    R0=risk(initial); V0=quantities(initial)[2]
    # ||log(x_init/x_center)||_2 <= sqrt(8)*delta/(1-delta) < 3/(den-1).
    E=I.rat(3,den-1); h=I.rat(1,stepden); duration=I(0,h.hi); start=time.time()
    result={'relative_half_width':f'1/{den}','bits':180,'step':f'1/{stepden}','taylor_order':order,'initial_risk':R0.out(),'initial_energy':V0.out(),'sections':{},'steps':[]}
    Hmin=None; xmin=None
    for step in range(6*stepden):
        f=field(x);tube=[]
        for k in range(6):
            spread=max(abs(f[k].lo),abs(f[k].hi))*h.hi//S+16
            tube.append(I(x[k].lo-16,x[k].hi+(spread*11+9)//10))
        tube+=x[6:]
        assert all(t.lo>0 for t in tube)
        ft=field(tube)
        assert all((x[k]+duration*ft[k]).inside(tube[k]) for k in range(6))
        # Find a rational logarithmic-norm bound, including its own error envelope.
        for n in range(1,81):
            m=I.rat(n,20)
            Enew=E/(ONE-m*h); r=I(Enew.hi)
            if r.hi>=S//2: continue
            factor=(ONE-r).hull(ONE/(ONE-r))
            wide=[v*factor for v in tube]
            if pd_bound(logjac(wide),m): break
        else: raise AssertionError('no verified logarithmic norm bound')
        E=I(Enew.hi)
        H=quantities(wide)[3]; assert H.lo>0
        Hmin=H.lo if Hmin is None else min(Hmin,H.lo)
        xmin=min(t.lo for t in wide) if xmin is None else min(xmin,min(t.lo for t in wide))
        cs=coeff(x,order-1); rem=coeff(tube,order)
        y=[]
        for k in range(6):
            val=rem[k][order]
            for j in reversed(range(order)): val=val*h+cs[k][j]
            y.append(val)
        x=y+x[6:]
        result['steps'].append({'m_numerator_over_20':n,'log_error_upper':str(E.hi),'H_lower':str(H.lo)})
        if step+1 in (17*stepden//10,6*stepden):
            factor=(ONE-E).hull(ONE/(ONE-E)); outer=[v*factor for v in x]
            R=risk(outer); key='1.7' if step+1==17*stepden//10 else '6'
            result['sections'][key]={'center_state':[v.out() for v in x],'outer_state':[v.out() for v in outer],'risk':R.out(),'difference_from_initial':(R-R0).out(),'log_error_upper':str(E.hi)}
            print('SECTION',key,'R',R.show(),'diff',(R-R0).show(),'E',E.show(),flush=True)
        if (step+1)%stepden==0: print('progress',(step+1)//stepden,'m',n/20,'E',E.show(),'seconds',round(time.time()-start,1),flush=True)
    r1=I(*map(int,result['sections']['1.7']['risk']));r2=I(*map(int,result['sections']['6']['risk']))
    assert (r1-R0).hi<=I.rat(-4,100).lo
    assert (r2-R0).lo>=I.rat(15,100).hi
    assert (r2-r1).lo>=I.rat(19,100).hi
    assert (R0-r1).lo>=I.rat(9122,100000).hi
    assert (r2-R0).lo>=I.rat(33664,100000).hi
    assert (r2-r1).lo>=I.rat(42790,100000).hi
    from fractions import Fraction as F
    t=F(1);e=t
    for k in range(1,41):t*=F(6,k);e+=t
    assert e+t*F(6,41)/(1-F(6,42))<404
    Q=V0*I.rat(36,25)*404;assert Q.hi<I.rat(689,1000).lo
    assert Hmin>I.rat(129,1000000).hi
    assert xmin>I.rat(3199,100000).hi
    result.update(status='PASS',H_lower=str(Hmin),tube_coordinate_lower=str(xmin),Q_upper=Q.out(),rebound=(r2-r1).out())
    return result
if __name__=='__main__':
    import argparse
    from pathlib import Path
    if not __debug__:
        raise RuntimeError('Certificate verification requires enabled assertions; do not use -O.')
    p=argparse.ArgumentParser();p.add_argument('--den',type=int,default=100000);p.add_argument('--output',default=str(result_path('certificate.json')));a=p.parse_args();assert a.den>1
    r=run(a.den)
    with open(a.output,'w') as f:json.dump(r,f,indent=2)
    print('PASS',a.output)

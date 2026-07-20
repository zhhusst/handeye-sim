#!/usr/bin/env python3
"""极简证明：plane_edge_9dof 方法行不行"""
import numpy as np, time
t0=time.time()

def se(w):
    t=np.linalg.norm(w)
    if t<1e-10: return np.eye(3)
    a=w/t; x,y,z=a; c,s=np.cos(t),np.sin(t)
    return np.array([[c+x*x*(1-c),x*y*(1-c)-z*s,x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s,c+y*y*(1-c),y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s,z*y*(1-c)+x*s,c+z*z*(1-c)]])
def sl(R):
    t=np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
    if t<1e-10: return np.zeros(3)
    w=(R-R.T)/(2*np.sin(t))
    return t*np.array([w[2,1],w[0,2],w[1,0]])

Rhe=se(np.deg2rad([12,-8,25])); the=np.array([.05,-.03,.12])
Rp=se(np.deg2rad([0,0,30])); ub,vb,nb=Rp[:,0],Rp[:,1],Rp[:,2]
Cp=np.array([.5,-.1,0]); pw,ph=.4,.5
tg=np.concatenate([sl(Rhe),the,sl(Rp)])

rng=np.random.default_rng(42)

def gen_meas(RBS,tBS):
    R_SB=RBS.T; t_SB=-R_SB@tBS; ln=RBS[:,1]; ld=np.cross(ln,nb)
    if np.linalg.norm(ld)<1e-10: return None
    ld/=np.linalg.norm(ld)
    A=np.vstack([ln.reshape(1,3),nb.reshape(1,3)])
    b=np.array([np.dot(ln,tBS),np.dot(nb,Cp)])
    P0=np.linalg.lstsq(A,b,rcond=None)[0]
    tv=np.linspace(-.3,.3,100); ii=[]
    for i,tv_ in enumerate(tv):
        pB=P0+tv_*ld; dp=pB-Cp; u,v=np.dot(dp,ub),np.dot(dp,vb)
        if 0<=u<=pw and 0<=v<=ph:
            pS=R_SB@pB+t_SB
            if .27<=pS[2]<=.82 and abs(pS[0])<=pS[2]*.2679: ii.append(i)
    if len(ii)<3: return None
    segs=[]; s=ii[0]
    for j in range(1,len(ii)):
        if ii[j]-ii[j-1]>1: segs.append((s,ii[j-1])); s=ii[j]
    segs.append((s,ii[-1])); bs=max(segs,key=lambda s:s[1]-s[0])
    idx=np.linspace(bs[0],bs[1],min(15,bs[1]-bs[0]+1),dtype=int)
    sc=np.array([R_SB@(P0+tv[i]*ld)+t_SB for i in idx])
    def ep(pt,dr,ex):
        d=np.dot(ln,dr)
        if abs(d)<1e-12: return None
        sv=np.dot(ln,tBS-pt)/d
        if -.002<=sv<=ex+.002:
            pB=pt+sv*dr; pS=R_SB@pB+t_SB
            if .27<=pS[2]<=.82 and abs(pS[0])<=pS[2]*.2679: return pS
        return None
    e1=[x for x in [ep(Cp,ub,pw),ep(Cp+ph*vb,ub,pw)] if x is not None]
    e2=[x for x in [ep(Cp,vb,ph),ep(Cp+pw*ub,vb,ph)] if x is not None]
    nm=0.008/1000; sc+=rng.normal(0,nm,sc.shape)
    e1=[p+rng.normal(0,nm,3) for p in e1]
    e2=[p+rng.normal(0,nm,3) for p in e2]
    return {'p_S_plane':sc,'p_S_e1':e1,'p_S_e2':e2}

def gen_pose(p,y,s,cv):
    target=Cp+cv[0]*ub+cv[1]*vb; zS=-nb
    ca=vb if abs(cv[1])<.05 else ub
    Ry=se(zS*np.deg2rad(y))
    xS=Ry@(ca/np.linalg.norm(ca))
    if abs(np.dot(xS,zS))>.999:
        xS=np.cross(zS,np.array([1.,0.,0.])); xS/=np.linalg.norm(xS)
    yS=np.cross(zS,xS); yS/=np.linalg.norm(yS); xS=np.cross(yS,zS)
    Rp_=se(xS*np.deg2rad(p))
    zSp=Rp_@zS; ySp=Rp_@yS; xSp=np.cross(ySp,zSp); xSp/=np.linalg.norm(xSp)
    ySp=np.cross(zSp,xSp)
    RS=np.column_stack([xSp,ySp,zSp]); tS=target+s*nb
    Ri=RS@Rhe.T; ti=tS-Ri@the
    m=gen_meas(Ri@Rhe,ti+Ri@the)
    if m is not None and len(m['p_S_plane'])>=3: return (Ri,ti,m)
    return None

PP,MM=[],[]
for is_e1 in [True]*6+[False]*6:
    cv=(rng.uniform(.05,pw-.05),rng.uniform(-.005,.005)) if is_e1 else \
       (rng.uniform(-.005,.005),rng.uniform(.05,ph-.05))
    for _ in range(15):
        res=gen_pose(rng.uniform(-15,15),rng.uniform(-25,25),rng.uniform(.40,.60),cv)
        if res: PP.append((res[0],res[1])); MM.append(res[2]); break

print(f"Poses: {len(PP)}")
print(f"e1 pts: {sum(len(m['p_S_e1']) for m in MM)}")
print(f"e2 pts: {sum(len(m['p_S_e2']) for m in MM)}")

def res(th,p,m):
    wh,th_,wp=th[:3],th[3:6],th[6:9]
    Rh,Rp_=se(wh),se(wp)
    u_,v_,n_=Rp_[:,0],Rp_[:,1],Rp_[:,2]
    pv,p1,p2=[],[],[]
    for(Ri,ti),mm in zip(p,m):
        RBS=Ri@Rh; tBS=ti+Ri@th_
        for ps in mm['p_S_e1']: p1.append(RBS@ps+tBS)
        for ps in mm['p_S_e2']: p2.append(RBS@ps+tBS)
        for ps in mm['p_S_plane']: pv.append(np.dot(n_,RBS@ps+tBS))
    pv=np.array(pv)
    if len(pv): pv-=np.mean(pv)
    r=[]
    for v in pv: r.append(v*np.sqrt(.1))
    for k in range(len(p1)-1): r.extend((np.cross(p1[k+1]-p1[k],u_)).tolist())
    for k in range(len(p2)-1): r.extend((np.cross(p2[k+1]-p2[k],v_)).tolist())
    return np.array(r)

def solve(ti,p,m):
    th=ti.copy(); lm=1e-6
    for _ in range(200):
        r0=res(th,p,m); J=np.zeros((len(r0),9)); eps=1e-6
        for j in range(9):
            tp=th.copy(); tp[j]+=eps; tm=th.copy(); tm[j]-=eps
            J[:,j]=(res(tp,p,m)-res(tm,p,m))/(2*eps)
        c0=.5*np.dot(r0,r0); H,g=J.T@J,J.T@r0
        try: d=-np.linalg.solve(H+lm*np.eye(9),g)
        except: lm*=10; continue
        tn=th+d; nc=.5*np.dot(res(tn,p,m),res(tn,p,m))
        if nc<c0: th=tn; lm=max(lm/3,1e-12)
        else: lm=min(lm*3,1e6)
        if abs(c0-nc)<1e-12: break
    return th

# P1: rank
J=np.zeros((len(res(tg,PP,MM)),9))
for j in range(9):
    tp=tg.copy(); tp[j]+=1e-6; tm=tg.copy(); tm[j]-=1e-6
    J[:,j]=(res(tp,PP,MM)-res(tm,PP,MM))/(2e-6)
s=np.linalg.svd(J,compute_uv=False)
print(f"\nP1: rank={np.sum(s>1e-10)}/9, s_min={s[-1]:.6f}, cond={s[0]/s[-1]:.0f}")

# P2: gauge
print(f"P2: continuous gauge? {'NO' if s[-1]>1e-12 else 'YES'}")
ths=tg.copy(); ths[6:9]=sl(se(np.array([0,0,np.pi]))@se(tg[6:9]))
c1=.5*np.dot(res(tg,PP,MM),res(tg,PP,MM))
c2=.5*np.dot(res(ths,PP,MM),res(ths,PP,MM))
print(f"P2: 180deg symmetry: dcost={c2-c1:.2e}")

# P3: convergence
th0=solve(np.zeros(9),PP,MM)
R0=np.rad2deg(np.arccos(np.clip((np.trace(se(th0[:3]).T@Rhe)-1)/2,-1,1)))
T0=np.linalg.norm(th0[3:6]-the)*1000
print(f"P3: zero-init: R={R0:.4f}deg, t={T0:.4f}mm")

bestR=999
for r in range(9):
    ti=np.zeros(9) if r==0 else np.concatenate([np.random.uniform(-2,2,3),np.zeros(3),np.random.uniform(-2,2,3)])
    th=solve(ti,PP,MM)
    R=np.rad2deg(np.arccos(np.clip((np.trace(se(th[:3]).T@Rhe)-1)/2,-1,1)))
    if R<bestR: bestR,bestT=R,np.linalg.norm(th[3:6]-the)*1000
print(f"P3: multi-restart 9x: R={bestR:.6f}deg, t={bestT:.4f}mm")

print(f"\nTime: {time.time()-t0:.1f}s")

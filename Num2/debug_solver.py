#!/usr/bin/env python3
"""Debug solver - standalone test"""
import numpy as np

def so3_exp(w):
    t=np.linalg.norm(w)
    if t<1e-10: return np.eye(3)
    a=w/t; x,y,z=a; c,s=np.cos(t),np.sin(t)
    return np.array([[c+x*x*(1-c),x*y*(1-c)-z*s,x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s,c+y*y*(1-c),y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s,z*y*(1-c)+x*s,c+z*z*(1-c)]])

def so3_log(R):
    t=np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
    if t<1e-10: return np.zeros(3)
    w=(R-R.T)/(2*np.sin(t))
    return t*np.array([w[2,1],w[0,2],w[1,0]])

# Truth
Rhe=so3_exp(np.deg2rad([12,-8,25]))
the=np.array([.05,-.03,.12])
Rp=so3_exp(np.deg2rad([0,0,30]))
ub,vb,nb=Rp[:,0],Rp[:,1],Rp[:,2]
C=np.array([.5,-.1,0]); pw,ph=.4,.5
tg=np.concatenate([so3_log(Rhe),the,so3_log(Rp)])

print(f"True theta: {tg}")

# === Generate measurements using forward geometry ===
def gen_meas(R_BS,t_BS,rng):
    """Generate measurement at given sensor pose"""
    R_SB=R_BS.T; t_SB=-R_SB@t_BS
    ln=R_BS[:,1]  # laser normal (y_S)
    ld=np.cross(ln,nb)
    if np.linalg.norm(ld)<1e-10: return None
    ld/=np.linalg.norm(ld)
    A=np.vstack([ln.reshape(1,3),nb.reshape(1,3)])
    b=np.array([np.dot(ln,t_BS),np.dot(nb,C)])
    P0=np.linalg.lstsq(A,b,rcond=None)[0]
    
    tv=np.linspace(-.3,.3,200)
    ok=[]
    for i,tv_ in enumerate(tv):
        pB=P0+tv_*ld; dp=pB-C
        u,v=np.dot(dp,ub),np.dot(dp,vb)
        if 0<=u<=pw and 0<=v<=ph:
            pS=R_SB@pB+t_SB
            if .27<=pS[2]<=.82 and abs(pS[0])<=pS[2]*.2679:
                ok.append(i)
    if len(ok)<3: return None
    
    segs=[]; s=ok[0]
    for j in range(1,len(ok)):
        if ok[j]-ok[j-1]>1: segs.append((s,ok[j-1])); s=ok[j]
    segs.append((s,ok[-1]))
    b_seg=max(segs,key=lambda s:s[1]-s[0])
    idx=np.linspace(b_seg[0],b_seg[1],min(20,b_seg[1]-b_seg[0]+1),dtype=int)
    scan_S=np.array([R_SB@(P0+tv[i]*ld)+t_SB for i in idx])
    
    def edge_pt(pt,dr,ex):
        d=np.dot(ln,dr)
        if abs(d)<1e-12: return None
        sv=np.dot(ln,t_BS-pt)/d
        if -.002<=sv<=ex+.002:
            pB=pt+sv*dr; pS=R_SB@pB+t_SB
            if .27<=pS[2]<=.82 and abs(pS[0])<=pS[2]*.2679: return pS
        return None
    
    e1=[edge_pt(C,ub,pw),edge_pt(C+ph*vb,ub,pw)]
    e2=[edge_pt(C,vb,ph),edge_pt(C+pw*ub,vb,ph)]
    e1=[x for x in e1 if x is not None]
    e2=[x for x in e2 if x is not None]
    
    nm=0.008/1000
    scan_S+=rng.normal(0,nm,scan_S.shape)
    e1=[p+rng.normal(0,nm,3) for p in e1]
    e2=[p+rng.normal(0,nm,3) for p in e2]
    return {'p_S_plane':scan_S,'p_S_e1':e1,'p_S_e2':e2}

def gen_pose(pitch_deg,yaw_deg,standoff_m,uv,rng):
    """Generate robot pose + measurement"""
    target=C+uv[0]*ub+uv[1]*vb
    zS=-nb; ca=vb if abs(uv[1])<.1 else ub
    
    Ry=so3_exp(zS*np.deg2rad(yaw_deg))
    xS=Ry@(ca/np.linalg.norm(ca))
    if abs(np.dot(xS,zS))>.999:
        xS=np.cross(zS,np.array([1.,0.,0.])); xS/=np.linalg.norm(xS)
    yS=np.cross(zS,xS); yS/=np.linalg.norm(yS); xS=np.cross(yS,zS)
    Rp=so3_exp(xS*np.deg2rad(pitch_deg))
    zSp=Rp@zS; ySp=Rp@yS; xSp=np.cross(ySp,zSp); xSp/=np.linalg.norm(xSp)
    ySp=np.cross(zSp,xSp)
    
    RS=np.column_stack([xSp,ySp,zSp]); tS=target+standoff_m*nb
    Ri=RS@Rhe.T; ti=tS-Ri@the
    RS2=Ri@Rhe; tS2=ti+Ri@the
    m=gen_meas(RS2,tS2,rng)
    if m is not None and len(m['p_S_plane'])>=5: return (Ri,ti,m)
    return None

# Generate 5+5 dataset
rng=np.random.default_rng(42)
poses,meas=[],[]
uvs=[(rng.uniform(.05,pw-.05),rng.uniform(-.005,.005)) for _ in range(5)] + \
    [(rng.uniform(-.005,.005),rng.uniform(.05,ph-.05)) for _ in range(5)]

for uv in uvs:
    for _ in range(15):
        res=gen_pose(rng.uniform(-15,15),rng.uniform(-25,25),rng.uniform(.40,.60),uv,rng)
        if res: poses.append((res[0],res[1])); meas.append(res[2]); break

print(f"Generated {len(poses)} poses")

# === Simple solver ===
def residuals(theta, poses, meas):
    w_h,t_h,w_p=theta[:3],theta[3:6],theta[6:9]
    Rh,Rp_=so3_exp(w_h),so3_exp(w_p)
    ub_,vb_,nb_=Rp_[:,0],Rp_[:,1],Rp_[:,2]
    pv,p1,p2=[],[],[]
    for(Ri,ti),m in zip(poses,meas):
        RBS=Ri@Rh; tBS=ti+Ri@t_h
        for p in m['p_S_e1']: p1.append(RBS@p+tBS)
        for p in m['p_S_e2']: p2.append(RBS@p+tBS)
        for p in m['p_S_plane']: pv.append(np.dot(nb_,RBS@p+tBS))
    pv=np.array(pv)
    if len(pv): pv-=np.mean(pv)
    r=[]; wp,we=np.sqrt(.1),np.sqrt(1.0)
    for v in pv: r.append(v*wp)
    for k in range(len(p1)-1):
        r.extend((np.cross(p1[k+1]-p1[k],ub_)*we).tolist())
    for k in range(len(p2)-1):
        r.extend((np.cross(p2[k+1]-p2[k],vb_)*we).tolist())
    return np.array(r)

def jac(theta,poses,meas,eps=1e-6):
    r0=residuals(theta,poses,meas)
    J=np.zeros((len(r0),9))
    for j in range(9):
        tp=theta.copy(); tp[j]+=eps
        tm=theta.copy(); tm[j]-=eps
        J[:,j]=(residuals(tp,poses,meas)-residuals(tm,poses,meas))/(2*eps)
    return J,r0

def solve(theta_init,poses,meas):
    theta=theta_init.copy(); lam=1e-6
    for it in range(200):
        J,r=jac(theta,poses,meas)
        c0=0.5*np.dot(r,r)
        H,g=J.T@J,J.T@r
        try: d=-np.linalg.solve(H+lam*np.eye(9),g)
        except: lam*=10; continue
        tn=theta+d
        rn=residuals(tn,poses,meas)
        nc=0.5*np.dot(rn,rn)
        if nc<c0: theta=tn; lam=max(lam/3,1e-12)
        else: lam=min(lam*3,1e6)
        if abs(c0-nc)<1e-12: 
            if it>0: pass # converged
            break
    return theta

# Test from zero init
th_e = solve(np.zeros(9), poses, meas)
Re,Te = so3_log(so3_exp(th_e[:3]).T @ Rhe), np.linalg.norm(th_e[3:6]-the)*1000
tr=np.clip((np.trace(so3_exp(th_e[:3]).T@Rhe)-1)/2,-1,1)
R_err=np.rad2deg(np.arccos(tr))

print(f"\nZero init:")
print(f"  R_err = {R_err:.4f}°")
print(f"  t_err = {np.linalg.norm(th_e[3:6]-the)*1000:.4f}mm")
print(f"  theta_est = {th_e}")
print(f"  theta_gt  = {tg}")

# Cost comparison
c_e=0.5*np.dot(residuals(th_e,poses,meas),residuals(th_e,poses,meas))
c_g=0.5*np.dot(residuals(tg,poses,meas),residuals(tg,poses,meas))
print(f"  cost_est = {c_e:.6e}")
print(f"  cost_gt  = {c_g:.6e}")

# Test from truth
th_t = solve(tg.copy(), poses, meas)
R_t = np.rad2deg(np.arccos(np.clip((np.trace(so3_exp(th_t[:3]).T@Rhe)-1)/2,-1,1)))
T_t = np.linalg.norm(th_t[3:6]-the)*1000
print(f"\nTruth init:")
print(f"  R_err = {R_t:.6f}°, t_err = {T_t:.6f}mm")
print(f"  theta from truth init: {th_t}")

# == SVD at truth ==
J,r=jac(tg,poses,meas)
s=np.linalg.svd(J,compute_uv=False)
print(f"\nSVD at truth: σ = {s}")
print(f"rank = {np.sum(s>1e-10)}/9, cond={s[0]/s[-1]:.0f}")

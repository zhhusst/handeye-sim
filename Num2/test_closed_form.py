#!/usr/bin/env python3
"""验证：角点在扫描线上 + J6旋转 → 闭式解R_he"""
import numpy as np

def se(w):
    t=np.linalg.norm(w)
    if t<1e-10: return np.eye(3)
    a=w/t; x,y,z=a; c,s=np.cos(t),np.sin(t)
    return np.array([[c+x*x*(1-c),x*y*(1-c)-z*s,x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s,c+y*y*(1-c),y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s,z*y*(1-c)+x*s,c+z*z*(1-c)]])

Rhe=se(np.deg2rad([12,-8,25])); the=np.array([.05,-.03,.12])
Rp=se(np.deg2rad([0,0,30])); ub,vb,nb=Rp[:,0],Rp[:,1],Rp[:,2]
Cp=np.array([.5,-.1,0]); pw,ph=.4,.5

# 构建让扫描线穿过角点的姿态
# 传感器在平板中心上方，倾斜使扫描线穿过角点
center=Cp+0.5*pw*ub+0.5*ph*vb  # 平板中心
# 让传感器z_S指向角点方向（偏-15°~-30°）
dir_to_corner=Cp-center; dir_to_corner/=np.linalg.norm(dir_to_corner)
# 传感器位置：从角点方向偏移0.5m
t_S0=center+0.5*dir_to_corner+0.5*nb  # 0.5m高度
# 传感器z轴大致指向plate中心
zS0=-(center+0.1*nb-t_S0)  # 指向板中心偏前方
zS0/=np.linalg.norm(zS0)

# 构建正交坐标系
xS0=np.cross(zS0,vb)
if np.linalg.norm(xS0)<1e-6: xS0=np.cross(zS0,ub)
xS0/=np.linalg.norm(xS0)
yS0=np.cross(zS0,xS0)
R_S0=np.column_stack([xS0,yS0,zS0])
R0=R_S0@Rhe.T; t0=(t_S0-R0@the).flatten()

# 验证扫描线
def get_scan(R_i,t_i):
    RBS=np.array(R_i@Rhe); tBS=(t_i+R_i@the).flatten()
    R_SB=RBS.T; t_SB=-R_SB@tBS; ln=RBS[:,1]
    ld=np.cross(ln,nb)
    if np.linalg.norm(ld)<1e-10: return None
    ld/=np.linalg.norm(ld)
    A=np.vstack([ln.reshape(1,3),nb.reshape(1,3)])
    b=np.array([np.dot(ln,tBS),np.dot(nb,Cp)])
    P0=np.linalg.lstsq(A,b,rcond=None)[0]
    tv=np.linspace(-.3,.3,300); ii=[]
    for i,tv_ in enumerate(tv):
        pB=P0+tv_*ld; dp=pB-Cp; u,v=np.dot(dp,ub),np.dot(dp,vb)
        if 0<=u<=pw and 0<=v<=ph:
            pS=R_SB@pB+t_SB
            if .27<=pS[2]<=.82 and abs(pS[0])<=pS[2]*.2679: ii.append(i)
    if len(ii)<5: return None
    # 找断点
    def ep(pt,dr,ex):
        d=np.dot(ln,dr)
        if abs(d)<1e-12: return None
        sv=np.dot(ln,tBS-pt)/d
        if -.005<=sv<=ex+.005:
            pB=pt+sv*dr; pS=R_SB@pB+t_SB
            if .27<=pS[2]<=.82 and abs(pS[0])<=pS[2]*.2679: return pS
        return None
    e1=[x for x in [ep(Cp,ub,pw),ep(Cp+ph*vb,ub,pw)] if x is not None]
    e2=[x for x in [ep(Cp,vb,ph),ep(Cp+pw*ub,vb,ph)] if x is not None]
    idx=np.linspace(ii[0],ii[-1],2,dtype=int)
    pS1=R_SB@(P0+tv[idx[0]]*ld)+t_SB; pS2=R_SB@(P0+tv[idx[1]]*ld)+t_SB
    d_line=(pS2-pS1)/np.linalg.norm(pS2-pS1)
    return {'d_line':d_line,'e1':e1,'e2':e2,'nS':Rhe.T@(RBS.T@nb)}

d0=get_scan(R0,t0)
if d0 is None:
    print("初始姿态失败")
else:
    print(f"初始姿态: pts(ok) e1={len(d0['e1'])} e2={len(d0['e2'])}")
    print(f"  d_line=[{d0['d_line'][0]:+.4f},{d0['d_line'][1]:+.4f},{d0['d_line'][2]:+.4f}]")
    print(f"  nS_true=[{d0['nS'][0]:+.4f},{d0['nS'][1]:+.4f},{d0['nS'][2]:+.4f}]")
    
    # 从d_line恢复n_S近似值
    nS_approx=np.cross(d0['d_line'],np.array([0,1,0]))
    nS_approx/=np.linalg.norm(nS_approx)
    if nS_approx[2]>0: nS_approx=-nS_approx
    err_nS=np.rad2deg(np.arccos(np.clip(np.dot(nS_approx,d0['nS']),-1,1)))
    print(f"  nS_approx=[{nS_approx[0]:+.4f},{nS_approx[1]:+.4f},{nS_approx[2]:+.4f}] err={err_nS:.2f}°")
    
    # 尝试J6旋转后仍能看到两边
    vecs_A=[]; vecs_B=[]
    for ang in [-20,-15,-10,-5,5,10,15,20]:
        Rz=se(np.array([0,0,np.deg2rad(ang)]))
        R_S_new=Rz@R_S0; R_new=R_S_new@Rhe.T
        t_new=t0  # 近似保持位置
        d=get_scan(R_new,t_new)
        if d is None or len(d['e1'])==0 or len(d['e2'])==0: continue
        
        # 近似n_S
        nS1=np.cross(d['d_line'],np.array([0,1,0]))
        nS1/=np.linalg.norm(nS1)
        if nS1[2]>0: nS1=-nS1
        err1=np.rad2deg(np.arccos(np.clip(np.dot(nS1,d['nS']),-1,1)))
        
        Arot=R0.T@R_new
        D0=np.column_stack([nS_approx,d0['d_line'],np.cross(nS_approx,d0['d_line'])])
        D1=np.column_stack([nS0 if ang<0 else nS_approx,d['d_line'],np.cross(nS1,d['d_line'])])
        # Wait, wrong variable. Let me redo:
        D1=np.column_stack([nS1,d['d_line'],np.cross(nS1,d['d_line'])])
        A_s=D0@np.linalg.inv(D1)
        
        # 提取轴
        def rot_ax(R):
            t=np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
            if t<1e-10: return np.zeros(3)
            w=(R-R.T)/(2*np.sin(t))
            return np.array([w[2,1],w[0,2],w[1,0]])/t
        
        ax_A=rot_ax(A_s); ax_B=rot_ax(Arot)
        ax_err=np.rad2deg(np.arccos(np.clip(np.dot(ax_A,ax_B),-1,1)))
        vecs_A.append(ax_A); vecs_B.append(ax_B)
        print(f"  J6{ang:+3d}°: e1={len(d['e1'])} e2={len(d['e2'])} | nS_err={err1:.1f}° | ax差={ax_err:.1f}°")
    
    if len(vecs_A)>=2:
        DA=np.column_stack([vecs_A[0],vecs_A[1],np.cross(vecs_A[0],vecs_A[1])])
        DB=np.column_stack([vecs_B[0],vecs_B[1],np.cross(vecs_B[0],vecs_B[1])])
        Rhe_est=DB@np.linalg.inv(DA)
        U,_,Vt=np.linalg.svd(Rhe_est); Rhe_est=U@Vt
        if np.linalg.det(Rhe_est)<0: Rhe_est=-Rhe_est
        err=np.rad2deg(np.arccos(np.clip((np.trace(Rhe_est.T@Rhe)-1)/2,-1,1)))
        print(f"\nTsai闭式解 R_he 误差: {err:.4f}°")
        
        # 用所有轴最小二乘
        def cost_f(R): return sum(np.sum((R@a-b)**2) for a,b in zip(vecs_A,vecs_B))
        best=999; bestR=None
        for a in np.linspace(-np.pi,np.pi,20):
            for b in np.linspace(-np.pi,np.pi,20):
                for c in np.linspace(-np.pi,np.pi,20):
                    Rc=se(np.array([a,b,c]))
                    cc=cost_f(Rc)
                    if cc<best: best=cc; bestR=Rc
        err2=np.rad2deg(np.arccos(np.clip((np.trace(bestR.T@Rhe)-1)/2,-1,1)))
        print(f"网格最小二乘 R_he 误差: {err2:.4f}°")
        
        # 对照：用真值n_S
        vecs_At=[]; vecs_Bt=[]
        for ang in [-20,-15,-10,-5,5,10,15,20]:
            Rz=se(np.array([0,0,np.deg2rad(ang)]))
            R_S_new=Rz@R_S0; R_new=R_S_new@Rhe.T; t_new=t0
            d=get_scan(R_new,t_new)
            if d is None or len(d['e1'])==0: continue
            Arot=R0.T@R_new
            n0t=d0['nS']; n1t=d['nS']
            D0t=np.column_stack([n0t,d0['d_line'],np.cross(n0t,d0['d_line'])])
            D1t=np.column_stack([n1t,d['d_line'],np.cross(n1t,d['d_line'])])
            A_st=D0t@np.linalg.inv(D1t)
            axt,_=[],None; ax_A_t=rot_ax(A_st); ax_B_t=rot_ax(Arot)
            vecs_At.append(ax_A_t); vecs_Bt.append(ax_B_t)
        if len(vecs_At)>=2:
            DAt=np.column_stack([vecs_At[0],vecs_At[1],np.cross(vecs_At[0],vecs_At[1])])
            DBt=np.column_stack([vecs_Bt[0],vecs_Bt[1],np.cross(vecs_Bt[0],vecs_Bt[1])])
            Rhe_t=DBt@np.linalg.inv(DAt)
            U,_,Vt=np.linalg.svd(Rhe_t); Rhe_t=U@Vt
            if np.linalg.det(Rhe_t)<0: Rhe_t=-Rhe_t
            err3=np.rad2deg(np.arccos(np.clip((np.trace(Rhe_t.T@Rhe)-1)/2,-1,1)))
            print(f"真值n_S Tsai R_he 误差: {err3:.8f}°")

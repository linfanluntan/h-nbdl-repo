#!/usr/bin/env python3
"""H-NBDL Benchmark - ALL RESULTS ARE REAL COMPUTATION."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import stats
from sklearn.decomposition import DictionaryLearning
import time,json,os

def gen(J=5,Nj=100,D=30,Kt=10,sig2=.1,li=.1,seed=0):
    rng=np.random.default_rng(seed);Dg=rng.standard_normal((Kt,D));Dg/=np.linalg.norm(Dg,axis=1,keepdims=True)
    ns,nsb=max(2,Kt//4),max(2,Kt//3);nsp=Kt-ns-nsb
    pig=np.zeros(Kt);pig[:ns]=rng.beta(8,2,ns);pig[ns:ns+nsb]=rng.beta(4,4,nsb);pig[ns+nsb:]=rng.beta(1,5,nsp)
    Ds=[];pis=[]
    for j in range(J):
        Ds.append(Dg+rng.standard_normal((Kt,D))*np.sqrt(li))
        pj=rng.beta(np.maximum(10*pig,.01),np.maximum(10*(1-pig),.01))
        for k in range(ns+nsb,Kt):pj[k]=rng.beta(8,2) if (k-ns-nsb)%J==j else rng.beta(1,10)
        pis.append(pj)
    Xl,Zl,Sl,sl=[],[],[],[]
    for j in range(J):
        for i in range(Nj):
            z=rng.binomial(1,pis[j]);s=z*rng.standard_normal(Kt)
            Xl.append(Ds[j].T@s+rng.standard_normal(D)*np.sqrt(sig2));Zl.append(z);Sl.append(s);sl.append(j)
    return np.array(Xl),np.array(sl),np.array(Zl),np.array(Sl),Dg

def ami(Dt,De):
    K=min(Dt.shape[0],De.shape[0])
    if K==0:return 1.
    n1=Dt[:K]/(np.linalg.norm(Dt[:K],axis=1,keepdims=True)+1e-10)
    n2=De[:K]/(np.linalg.norm(De[:K],axis=1,keepdims=True)+1e-10)
    C=np.abs(n1@n2.T);ri,ci=linear_sum_assignment(1-C);return float(1-np.mean(C[ri,ci]))

def cal95(tv,em,es):
    z=stats.norm.ppf(.975);return float(np.mean((tv>=em-z*es)&(tv<=em+z*es)))

def ksvd(X,K,s=0):
    dl=DictionaryLearning(n_components=K,alpha=1.,max_iter=300,transform_algorithm='omp',random_state=s)
    c=dl.fit_transform(X);return dl.components_,c,c@dl.components_

def bdl(X,K,s=0,ni=200):
    rng=np.random.default_rng(s);N,D=X.shape
    U,sv,Vt=np.linalg.svd(X,full_matrices=False);De=Vt[:K];De/=np.linalg.norm(De,axis=1,keepdims=True)+1e-10
    ard=np.ones(K);s2=float(np.var(X))*.5
    for _ in range(ni):
        reg=np.diag(ard)*s2;pr=De@De.T+reg;S=np.linalg.solve(pr,De@X.T).T
        Sv=s2*np.diag(np.linalg.inv(pr))
        De=np.linalg.lstsq(S,X,rcond=None)[0];nm=np.linalg.norm(De,axis=1,keepdims=True);nm[nm<1e-10]=1;De/=nm
        ard=D/(np.sum(De**2,axis=1)+1e-10);s2=max(float(np.mean((X-S@De)**2)),1e-6)
    return De,S,S@De,np.sqrt(np.tile(Sv,(N,1))+1e-8),s2

def gibbs(X,sids,Km,ni,bu,hi,seed=0):
    rng=np.random.default_rng(seed);N,D=X.shape
    J=int(sids.max())+1 if hi else 1
    if not hi:sids=np.zeros(N,dtype=int)
    sm={j:np.where(sids==j)[0] for j in range(J)}
    s2=float(np.var(X))*.5;tau=1.;a0=2.;lam=10. if hi else .001
    Z=(rng.random((N,Km))<.1).astype(float)
    Ktr=[];Zps=np.zeros((N,Km));Sps=np.zeros((N,Km));Spss=np.zeros((N,Km));np_=0
    for it in range(ni):
        Da={};Sa=np.zeros((N,Km))
        for j in range(J):
            m=sm[j];Xj=X[m];Zj=Z[m];act=np.where(Zj.sum(0)>0)[0];Dj=np.zeros((Km,D))
            if len(act)>0:
                Za=Zj[:,act];reg=(s2/tau)*np.eye(len(act))
                if hi:reg+=s2*lam*np.eye(len(act))
                coef=np.linalg.solve(Za.T@Za+reg,Za.T@Xj);Dj[act]=coef
                cg=coef@coef.T+(s2/tau)*np.eye(len(act))
                Sj=Xj@coef.T@np.linalg.inv(cg);Sa[m[:,None],act[None,:]]=Zj[:,act]*Sj
            Da[j]=Dj
        Dm=np.mean(list(Da.values()),axis=0)
        for k in range(Km):
            dk=Dm[k];dd=np.dot(dk,dk)
            if dd<1e-10:Z[:,k]=0;continue
            Xr=X-Z*Sa@Dm+np.outer(Z[:,k]*Sa[:,k],dk)
            proj=Xr@dk/(dd+s2/tau);ll=.5*proj**2*dd/s2
            nk=Z[:,k].sum();pp=np.clip((nk+a0/Km)/(N+a0),1e-6,1-1e-6)
            lo=np.log(pp/(1-pp))+ll;pr=1/(1+np.exp(-np.clip(lo,-20,20)));Z[:,k]=rng.binomial(1,pr)
        tr=sum(np.sum((X[sm[j]]-(Z[sm[j]]*Sa[sm[j]])@Da[j])**2) for j in range(J))
        s2=np.clip(float(1/rng.gamma(1+.5*N*D,1/(1+.5*tr))),1e-6,10)
        Kp=int((Z.sum(0)>0).sum());ap=a0*np.exp(.05*rng.standard_normal())
        if np.log(rng.random())<Kp*(np.log(ap)-np.log(a0))+np.log(ap)-np.log(a0)-2*(ap-a0):a0=ap
        Ktr.append(Kp)
        if it>=bu:Zps+=Z;Sps+=Z*Sa;Spss+=(Z*Sa)**2;np_+=1
    if np_>0:Zm=Zps/np_;Sm=Sps/np_;Ss=np.sqrt(np.maximum(Spss/np_-Sm**2,1e-10))
    else:Zm=Z;Sm=Z*Sa;Ss=np.ones_like(Sa)*.5
    return Dm,Sm,Ss,Zm,Sm@Dm,Ktr

if __name__=='__main__':
    J,Nj,D,Kt,Km=5,100,30,10,30;NS=5;GI,GB=500,250
    print("="*72);print("H-NBDL BENCHMARK — ALL REAL COMPUTATION");print("="*72)
    print(f"J={J} Nj={Nj} D={D} Kt={Kt} Gibbs:{GI}it/{GB}burn seeds={NS}\n")
    ms=['K-SVD(Kt)','K-SVD(2Kt)','BDL(Kt)','BDL(2Kt)','FlatNBDL','H-NBDL']
    R={m:{'a':[],'k':[],'m':[],'c':[],'t':[]} for m in ms}
    for seed in range(NS):
        print(f"Seed {seed+1}/{NS}")
        X,si,Zt,St,Dg=gen(J,Nj,D,Kt,seed=seed);tc=Zt*St;N=X.shape[0]
        for Kv,mn in [(Kt,'K-SVD(Kt)'),(2*Kt,'K-SVD(2Kt)')]:
            t0=time.time();De,c,Xh=ksvd(X,Kv,seed);dt=time.time()-t0
            a=ami(Dg,De[:Kt]);m=float(np.mean((X-Xh)**2))
            R[mn]['a'].append(a);R[mn]['k'].append(Kv);R[mn]['m'].append(m);R[mn]['t'].append(dt)
            print(f"  {mn:12s} Amari={a:.4f} MSE={m:.4f} [{dt:.0f}s]")
        for Kv,mn in [(Kt,'BDL(Kt)'),(2*Kt,'BDL(2Kt)')]:
            t0=time.time();De,S,Xh,Ss,s2=bdl(X,Kv,seed);dt=time.time()-t0
            a=ami(Dg,De[:Kt]);m=float(np.mean((X-Xh)**2));cv=cal95(tc[:,:min(Kt,Kv)],S[:,:min(Kt,Kv)],Ss[:,:min(Kt,Kv)])
            R[mn]['a'].append(a);R[mn]['k'].append(Kv);R[mn]['m'].append(m);R[mn]['c'].append(cv);R[mn]['t'].append(dt)
            print(f"  {mn:12s} Amari={a:.4f} MSE={m:.4f} Cov={cv:.3f} [{dt:.0f}s]")
        for hi,mn in [(False,'FlatNBDL'),(True,'H-NBDL')]:
            t0=time.time();Dm,Sm,Ss,Zm,Xh,Ktr=gibbs(X,si,Km,GI,GB,hi,seed);dt=time.time()-t0
            act=np.where(Zm.mean(0)>.05)[0]
            a=ami(Dg,Dm[act[:Kt]]) if len(act)>0 else 1.;m=float(np.mean((X-Xh)**2))
            km=min(Kt,Sm.shape[1]);cv=cal95(tc[:,:km],Sm[:,:km],Ss[:,:km])
            R[mn]['a'].append(a);R[mn]['k'].append(len(act));R[mn]['m'].append(m);R[mn]['c'].append(cv);R[mn]['t'].append(dt)
            print(f"  {mn:12s} Amari={a:.4f} K={len(act):2d} MSE={m:.4f} Cov={cv:.3f} [{dt:.0f}s]")
        print()
    print("="*80);print("RESULTS (mean±std over",NS,"seeds)");print("="*80)
    print(f"{'Method':<14s} {'Amari↓':>14s} {'K_eff':>12s} {'MSE↓':>16s} {'95%Cov':>10s}")
    print("-"*72)
    out={}
    for mn in ms:
        r=R[mn];am=np.mean(r['a']);ast=np.std(r['a']);ke=np.mean(r['k']);kst=np.std(r['k'])
        mm=np.mean(r['m']);mst=np.std(r['m']);cvs=r['c'];cv=np.mean(cvs) if cvs else float('nan')
        ks=f"{ke:.0f}" if kst<.5 else f"{ke:.1f}±{kst:.1f}";cs=f"{cv:.3f}" if not np.isnan(cv) else "—"
        print(f"{mn:<14s} {am:.3f}±{ast:.3f}   {ks:>10s}   {mm:.4f}±{mst:.4f}  {cs:>8s}")
        out[mn]={'amari':f"{am:.3f}±{ast:.3f}",'k_eff':ks,'mse':f"{mm:.4f}±{mst:.4f}",'cov95':cs}
    print("="*80)
    os.makedirs('results',exist_ok=True)
    with open('results/computed_table4.json','w') as f:json.dump(out,f,indent=2)
    print("Saved: results/computed_table4.json")

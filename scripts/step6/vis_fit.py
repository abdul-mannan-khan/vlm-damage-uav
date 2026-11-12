import json,glob,numpy as np,collections
exec(open('scripts/step6/vis_check.py').read().split("print('T  |")[0].replace("print('p(d)","0 and print('p(d)"))
rng=np.random.default_rng(0)
Ts=(30,60,90,120,180,300)
# precompute per mission per T: list of per-building h-values per frame (flown poses) and twin
def hseq(K,poses):  # poses list of (t,p)
  return [(t,pd(np.hypot(*(K-p).T))) for t,p in poses]
F=[]
for run,(r,K,fr) in data.items():
  kin=np.array(json.load(open(run+'tracks_kin_fixed.json'))['kinematic'])
  fl=hseq(K,[(t,p) for t,u,p,s in fr]); tw=hseq(K,[(t,p[:2]) for t in range(0,300,5) for p in kin[t]])
  obs={T:np.zeros(len(K),bool) for T in Ts}
  for t,u,p,s in fr:
    for T in Ts:
      if t<=T: obs[T]|=s
  F.append((r,fl,tw,{T:obs[T].mean() for T in Ts}))
V=rng.gamma(1,1,size=(400,1))  # placeholder
def pred(seq,T,k):
  v=rng.gamma(k,1/k,size=(200,1)); logm=0
  for t,h in seq:
    if t<=T: logm=logm+np.log(np.clip(1-np.minimum(1,v*h[None,:]),1e-12,1))
  return (1-np.exp(logm)).mean()
best=None
for k in (0.3,0.5,0.7,1,1.5,2,3,5):
  err=np.mean([(pred(fl,T,k)-o[T])**2 for r,fl,tw,o in F for T in Ts])
  print('k',k,'rmse',round(err**.5,3))
  if best is None or err<best[0]: best=(err,k)
k=best[1];print('best k',k)
print('T flown predflown predtwin')
for T in Ts:
  print(T, round(np.mean([o[T] for *_,o in F]),2), round(np.mean([pred(fl,T,k) for r,fl,tw,o in F]),2), round(np.mean([pred(tw,T,k) for r,fl,tw,o in F]),2))
json.dump(dict(k=k,pc=pc.tolist(),bin_m=10),open('build/vis_model.json','w'))
# MAQAC-S - RAG predicted and flown by T
for T in (30,60,90,120,300):
  d=[];dp=[]
  for n in (4,6):
    for s in range(20000,20004):
      A=[x for x in F if x[0]['n_uav']==n and x[0]['seed']==s]
      a=[x for x in A if x[0]['method']=='maqac_seq'][0];b=[x for x in A if x[0]['method']=='rag'][0]
      d.append(a[3][T]-b[3][T]); dp.append(pred(a[2],T,k)-pred(b[2],T,k))
  print('T',T,'MAQAC-S - RAG flown %+.3f  calibrated-twin %+.3f'%(np.mean(d),np.mean(dp)), 'flown pos/8',sum(x>0 for x in d))

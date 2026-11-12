import json,glob,numpy as np,collections
D='results/vlm_damage/city/';B=json.load(open(D+'buildings.json'))
fp={b['name']:(round(b['N']),round(b['E'])) for b in B}
edges=np.arange(0,301,10); hit=np.zeros(30); tot=np.zeros(30)
runs=sorted(glob.glob(D+'runs/vlm_*/'))
def frames(run,keys):
  out=[]
  for l in open(run+'frames/labels.jsonl'):
    x=json.loads(l);acc=collections.Counter()
    for n,p in x['bld'] or []:
      if n in fp: acc[fp[n]]+=p
    out.append((x['t'],x['uav'],np.array(x['pose_ned'][:2]),np.array([acc.get(k,0)>=6144 for k in keys])))
  return out
data={}
for run in runs:
  r=json.load(open(run+'result.json'));o,L=r['origin_ne'],r['L']
  keys=sorted({fp[b['name']] for b in B if abs(b['N']-o[0])<=L/2 and abs(b['E']-o[1])<=L/2})
  K=np.array(keys,float); fr=frames(run,keys); data[run]=(r,K,fr)
  for t,u,p,s in fr:
    d=np.hypot(*(K-p).T); b=(d//10).astype(int); m=b<30
    np.add.at(tot,b[m],1); np.add.at(hit,b[m],s[m])
pc=hit/np.maximum(tot,1)
print('p(d) 10 m bins:',np.round(pc,2).tolist())
def pd(d): return np.where(d<300,pc[np.minimum((d//10).astype(int),29)],0)
# predictions: (a) flown frame poses, independence; (b) twin track every 5 s
print('T  | flown | pred(flown poses) | pred(twin track) | disc58(twin)')
for T in (30,60,90,120,180,300):
  f=[];a=[];b=[];c=[]
  for run,(r,K,fr) in data.items():
    seen=np.zeros(len(K),bool);miss=np.ones(len(K))
    for t,u,p,s in fr:
      if t<=T: seen|=s; miss*=1-pd(np.hypot(*(K-p).T))
    kin=np.array(json.load(open(run+'tracks_kin_fixed.json'))['kinematic'])
    m2=np.ones(len(K));dk=np.zeros(len(K),bool)
    for t in range(0,min(T,len(kin)-1)+1,5):
      for p in kin[t]:
        dd=np.hypot(*(K-p[:2]).T); m2*=1-pd(dd); dk|=dd<=58
    f.append(seen.mean());a.append(1-miss.mean());b.append(1-m2.mean());c.append(dk.mean())
  print(f'{T:3d} | {np.mean(f):.2f} | {np.mean(a):.2f} | {np.mean(b):.2f} | {np.mean(c):.2f}')
np.save('build/pc.npy',pc)

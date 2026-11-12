import json,sys,time,numpy as np
from pathlib import Path
ROOT=Path('.').resolve()
sys.path.insert(0,str(ROOT/'sim/airsim_vlm/netsim'));sys.path.insert(0,str(ROOT/'sim/airsim_vlm/airsim/maqac_px4'))
from sim import realistic; from episode import run_episode; from heightmap import AreaFrame
vm=json.load(open('build/vis_model.json'));pc=np.array(vm['pc']);k=vm['k']
B=json.load(open('results/vlm_damage/city/buildings.json'));o=[40.0,-10.0];L=600.0
K=np.array(sorted({(round(b['N']),round(b['E'])) for b in B if abs(b['N']-o[0])<=L/2 and abs(b['E']-o[1])<=L/2}),float)
def pd(d): return np.where(d<300,pc[np.minimum((d//10).astype(int),29)],0)
rng=np.random.default_rng(1);v=rng.gamma(k,1/k,size=(400,1))
def pred(tr,T):
  lm=np.zeros((400,len(K)))
  for t in range(0,min(T,len(tr)-1)+1,5):
    for p in tr[t]: lm+=np.log(np.clip(1-np.minimum(1,v*pd(np.hypot(*(K-np.array(p)).T))[None]),1e-12,1))
  return (1-np.exp(lm)).mean()
res={}
t0=time.time()
for n in (2,3):
  s=(L/1000)**2
  cfg=realistic(name=f'disaster_{n}uav',scenario='disaster',n_uav=n,T=300,perception='onboard',altitude=100.0,world=L,gateway=(L/2,L/2),n_users=int(round(200*s)),arrival_rate=0.1*s,takeoff=False)
  fr=AreaFrame(cfg.world,o)
  for m in ('maqac_seq','rag','maqac'):
    for sd in range(20000,20008):
      tr=[];run_episode(cfg,m,sd,step_hook=lambda W,a,st:tr.append(fr.to_world(W.upos[:,:2]).tolist()))
      res[(n,m,sd)]={T:pred(tr,T) for T in (60,120,180,300)}
  print('n',n,'done',round(time.time()-t0),'s',flush=True)
for n in (2,3):
  for m in ('maqac_seq','rag','maqac'):
    print(n,m,{T:round(np.mean([res[(n,m,s)][T] for s in range(20000,20008)]),2) for T in (60,120,180,300)})
  d={T:[res[(n,'maqac_seq',s)][T]-res[(n,'rag',s)][T] for s in range(20000,20008)] for T in (60,120,180,300)}
  print('  MAQAC-S - RAG',{T:(round(np.mean(x)*len(K),1),sum(y>0 for y in x)) for T,x in d.items()})

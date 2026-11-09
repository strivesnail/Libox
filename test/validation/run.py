"""python3 test/validation/run.py; compare fixed checkout and pristine sibling."""
import csv,json,os,pathlib,statistics,subprocess
root=pathlib.Path(__file__).resolve().parents[2]
p=root/'test/validation/results';p.mkdir(exist_ok=True)
baseline=root.parent/'Libox-baseline'
cpu=min(os.sched_getaffinity(0))
def build(src,output,include,sanitize=False):
 flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitize else ['-O3']
 subprocess.run(['g++','-std=c++20','-DNDEBUG','-DLIBOX','-march=native','-fopenmp','-pthread',*flags,'-I'+str(include/'src'),str(src),'-o',str(output)],check=True)
def run(cmd,name,timeout=30,env=None):
 with (p/(name+'.stdout')).open('w') as out,(p/(name+'.stderr')).open('w') as err:
  try:r=subprocess.run(['taskset','-c',str(cpu),*map(str,cmd)],stdout=out,stderr=err,timeout=timeout,env=env);code=r.returncode
  except subprocess.TimeoutExpired:code='timeout'
 return code
n=524288;domain=n*4096
# Exactly 32 initial and 64 final keys per logical slot: a stable, non-splitting
# control workload. Identical final key sets for both versions.
conf=p/'performance.csv'
conf.write_text(''.join(f'{i*domain//8},{(i+1)*domain//8},131072\n' for i in range(8)))
functional_conf=p/'functional.csv';d=8192*4096
functional_conf.write_text(''.join(f'{i*d//8},{(i+1)*d//8},196608\n' for i in range(8)))
statuses=[]
for version,checkout in [('baseline',baseline),('fixed',root)]:
 build(root/'test/regression.cpp',p/(version+'_functional'),checkout)
 for case in ['overflow','deleted_export','mixed','parallel']:
  code=run([p/(version+'_functional'),case,functional_conf],version+'_'+case)
  statuses.append((version,case,code));print(version,case,code,flush=True)
 build(root/'test/validation/performance.cpp',p/(version+'_performance'),checkout)
build(root/'test/regression.cpp',p/'fixed_asan',root,True)
env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0',UBSAN_OPTIONS='halt_on_error=1')
for case in ['overflow','deleted_export','mixed','parallel']:
 code=run([p/'fixed_asan',case,functional_conf],'asan_'+case,env=env)
 statuses.append(('fixed_asan',case,code));print('asan',case,code,flush=True)
# Dynamic split regression and per-stage throughput. Baseline failing runs
# are recorded as failures, never counted as performance wins.
for version,checkout in [('baseline',baseline),('fixed',root)]:
 build(root/'test/validation/hotspot.cpp',p/(version+'_hotspot'),checkout)
 for count in [32768,131072]:
  domain=count*4096;hotconf=p/f'hotspot_{count}.csv'
  hotconf.write_text(''.join(f'{i*domain//8},{(i+1)*domain//8},196608\n' for i in range(8)))
  for seed in [42,43,44]:
   for mode in ['hot_random','hot_sorted']:
    name=f'hotspot_{version}_{count}_{seed}_{mode}'
    code=run([p/(version+'_hotspot'),mode,hotconf,seed,count],name,20)
    statuses.append((version,name,code));print(name,code,flush=True)
rows=[]
for repeat in range(7):
 for version in (['baseline','fixed'] if repeat%2==0 else ['fixed','baseline']):
  name=f'perf_{repeat}_{version}'
  code=run([p/(version+'_performance'),conf,100+repeat],name,60)
  statuses.append((version,f'perf_{repeat}',code));print(name,code,flush=True)
  if code==0:
   for line in (p/(name+'.stderr')).read_text().splitlines():
    if line.startswith('PERF,'):
     _,op,rate=line.split(',');rows.append((repeat,version,op,float(rate)))
with (p/'performance_raw.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['repeat','version','operation','Mop_s']);w.writerows(rows)
summary=[]
for op in ['insert','lookup_hit','lookup_miss','update','delete']:
 a=[r[3] for r in rows if r[1]=='baseline' and r[2]==op];b=[r[3] for r in rows if r[1]=='fixed' and r[2]==op]
 if len(a)==len(b)==7:
  ma,mb=statistics.median(a),statistics.median(b)
  summary.append(dict(operation=op,baseline_median=ma,fixed_median=mb,change_pct=(mb/ma-1)*100,baseline_min=min(a),baseline_max=max(a),fixed_min=min(b),fixed_max=max(b)))
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(p/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
(p/'metadata.txt').write_text('CPU pinned: '+str(cpu)+'\nCompiler: '+subprocess.check_output(['g++','--version'],text=True).splitlines()[0]+'\nBaseline commit: '+subprocess.check_output(['git','-C',str(baseline),'rev-parse','HEAD'],text=True)+'\nFixed diff:\n'+subprocess.check_output(['git','-C',str(root),'diff'],text=True))
print(json.dumps(summary,indent=2))

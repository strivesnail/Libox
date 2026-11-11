"""Point-operation regression matrix. --quick reduces seeds; --tsan adds race checks."""
from pathlib import Path
import argparse,subprocess,os,json
parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');parser.add_argument('--tsan',action='store_true');args=parser.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'build/extended-checks';out.mkdir(parents=True,exist_ok=True);is_libox=(root/'src/libox.h').exists();flags=['-std=c++20','-O1','-g','-march=native','-fopenmp','-pthread','-I'+str(root/'src'),'-I'+str(root),*(['-DLIBOX'] if is_libox else [])]
for src,name in [('extended_regression.cpp','extended'),('concurrent_value_regression.cpp','snapshot'),('version_wrap_regression.cpp','version_wrap'),('typed_box_regression.cpp','typed_box')]:subprocess.run(['g++',*flags,'-fsanitize=address,undefined','-fno-omit-frame-pointer',str(root/'test'/src),'-o',str(out/name)],check=True)
cpus=sorted(os.sched_getaffinity(0));rows=[]
def run(name,binary,argv,threads,tsan=False):
 env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0',UBSAN_OPTIONS='halt_on_error=1',TSAN_OPTIONS='halt_on_error=1',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='TRUE',OMP_PLACES=','.join('{'+str(c)+'}' for c in cpus[:threads]))
 cmd=['taskset','-c',','.join(map(str,cpus[:threads])),str(out/binary),*map(str,argv)]
 if tsan:cmd=['setarch','x86_64','-R',*cmd]
 with (out/(name+'.log')).open('w') as log:
  try:r=subprocess.run(cmd,env=env,stdout=log,stderr=log,timeout=90);code=r.returncode
  except subprocess.TimeoutExpired:code='timeout'
 rows.append({'test':name,'exit':code});(out/'status.json').write_text(json.dumps(rows,indent=2));print(name,code,flush=True)
 if code:raise SystemExit('failed '+name)
for extra in ['version_wrap','typed_box']:run(extra,extra,[],1)
for mode in ['segment_bounds','split_admission','duplicate_index','empty_config','multiple_indexes','invalid_threads','empty_bulk','bulk_duplicates']:run(mode,'extended',[mode,5 if mode in ['multiple_indexes','bulk_duplicates'] else 1,42],5 if mode in ['multiple_indexes','bulk_duplicates'] else 1)
for mode in ['box_reference','segment_reference','random_index']:
 for seed in ([42,43] if args.quick else range(42,54)):run(f'{mode}_{seed}','extended',[mode,1,seed],1)
for nt in [n for n in [1,2,5,10,24] if n<=len(cpus)]:
 for seed in ([42] if args.quick else [42,43,44,45]):run(f'duplicates_t{nt}_s{seed}','extended',['concurrent_duplicate',nt,seed],nt)
for level in ['box','index']:
 for writers in [1,2,4]:run(f'snapshot_{level}_w{writers}','snapshot',[10000 if args.quick else 50000,writers,level],min(writers+1,len(cpus)))
if args.tsan:
 subprocess.run(['g++',*flags,'-fsanitize=thread','-fno-pie','-no-pie',str(root/'test/concurrent_value_regression.cpp'),'-o',str(out/'snapshot_tsan')],check=True)
 for level in ['box','index']:
  for writers in [1,2,4]:run(f'tsan_{level}_w{writers}','snapshot_tsan',[10000 if args.quick else 20000,writers,level],min(writers+1,len(cpus)),True)

"""Build and run bounded concurrent split regressions (requires AVX-512/OpenMP).
Usage: python3 test/run_concurrency_checks.py [--release] [--quick]
Build products and logs are written under build/concurrency-checks/.
"""
import argparse,json,os,pathlib,subprocess
parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true');parser.add_argument('--quick',action='store_true');args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];out=root/'build/concurrency-checks';out.mkdir(parents=True,exist_ok=True)
is_libox=(root/'src/libox.h').exists();cxx=os.environ.get('CXX','g++')
flags=['-std=c++20','-march=native','-fopenmp','-pthread','-I'+str(root/'src'),'-I'+str(root)]
flags+=['-O3','-DNDEBUG'] if args.release else ['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
def build(source,binary,extra=()):subprocess.run([cxx,*flags,*extra,str(root/source),'-o',str(out/binary)],check=True)
build('test/concurrent_gate_regression.cpp','gate')
build('test/segment_bounds_regression.cpp','bounds',['-DLIBOX'] if is_libox else [])
# Small fanout is deliberate: exercise recursive inner splits, not just buckets.
build('test/concurrent_split_regression.cpp','split',['-DLIBOX'] if is_libox else ['-DSPLIT_THRESHOLD=8'])
env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0',UBSAN_OPTIONS='halt_on_error=1',OMP_PROC_BIND='TRUE',OMP_DYNAMIC='FALSE')
cpus=sorted(os.sched_getaffinity(0));rows=[]
def run(name,cmd,threads):
 current=dict(env,OMP_PLACES=','.join('{'+str(c)+'}' for c in cpus[:threads]))
 with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
  try:r=subprocess.run(['taskset','-c',','.join(map(str,cpus[:threads])),*map(str,cmd)],stdout=stdout,stderr=stderr,env=current,timeout=45);code=r.returncode
  except subprocess.TimeoutExpired:code='timeout_45s'
 rows.append(dict(test=name,exit=code));(out/'status.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,code,flush=True)
 if code!=0:raise SystemExit(f'Failed: {name}; see {out}')
run('gate',[out/'gate'],min(4,len(cpus)))
run('bounds',[out/'bounds'],1)
for threads in [t for t in [1,4,24] if t<=len(cpus)]:
 for seed in ([42] if args.quick else [42,43,44]):
  for order in ['random','sorted']:
   run(f'split_t{threads}_s{seed}_{order}',[out/'split',threads,seed,order],threads)

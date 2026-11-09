#include <bits/stdc++.h>
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT=INT64_MAX;
using namespace liboxns;
using K=int64_t;using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){
 const size_t n=524288;const K domain=n*4096;
 uint64_t seed=std::stoull(argv[2]);std::mt19937_64 rng(seed);
 LiBox<K,K> index(.5,.1,1);index.loadConfigByFile(argv[1]);
 std::vector<K> keys,added;keys.reserve(n*2);added.reserve(n);
 for(size_t i=0;i<n;i++){K k=i*4096;keys.push_back(k);if(index.insertKeyValue(k,k+77).status!=InsertStatus::SUCCESS)return 2;}
 for(size_t i=0;i<n;i++)added.push_back(i*4096+2048);
 std::shuffle(added.begin(),added.end(),rng);
 auto t=Clock::now();size_t good=0;
 for(K k:added)good+=index.insertKeyValue(k,k+77).status==InsertStatus::SUCCESS;
 auto emit=[&](const char* name,size_t ops,Clock::time_point start){double secs=std::chrono::duration<double>(Clock::now()-start).count();std::cerr<<"PERF,"<<name<<","<<ops/secs/1e6<<"\n";};
 emit("insert",n,t);if(good!=n)return 3;
 keys.insert(keys.end(),added.begin(),added.end());std::shuffle(keys.begin(),keys.end(),rng);
 // Warm up outside timing; verify values after all inserts.
 for(K k:keys){auto r=index.searchKey(k);if(r.status!=SearchStatus::SUCCESS||r.value!=k+77)return 4;}
 uint64_t checksum=0;t=Clock::now();
 for(int pass=0;pass<4;pass++)for(K k:keys){auto r=index.searchKey(k);checksum+=uint64_t(r.value);good+=r.status==SearchStatus::SUCCESS;}
 emit("lookup_hit",keys.size()*4,t);
 if(good!=n+keys.size()*4)return 5;
 good=0;t=Clock::now();
 for(int pass=0;pass<4;pass++)for(K k:keys)good+=index.searchKey(k+1).status==SearchStatus::NOT_FOUND;
 emit("lookup_miss",keys.size()*4,t);if(good!=keys.size()*4)return 6;
 good=0;t=Clock::now();for(K k:keys)good+=index.insertKeyValue(k,-k-77).status==InsertStatus::SUCCESS;
 emit("update",keys.size(),t);if(good!=keys.size())return 7;
 for(K k:keys){auto r=index.searchKey(k);if(r.status!=SearchStatus::SUCCESS||r.value!=-k-77)return 8;}
 good=0;t=Clock::now();for(size_t i=0;i<keys.size();i+=2)good+=index.deleteKey(keys[i]).found;
 emit("delete",keys.size()/2,t);if(good!=keys.size()/2)return 9;
 for(size_t i=0;i<keys.size();i++){auto r=index.searchKey(keys[i]);if(i%2?(r.status!=SearchStatus::SUCCESS||r.value!=-keys[i]-77):r.status==SearchStatus::SUCCESS)return 10;}
 std::cerr<<"PASS checksum="<<checksum<<"\n";
}

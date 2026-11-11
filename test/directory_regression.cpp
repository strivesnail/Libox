#include <bits/stdc++.h>
#include <omp.h>
#include <unistd.h>
#define private public
#include "libox.h"
#undef private
size_t MAX_KEYS_IN_SEGMENT=INT64_MAX;
using K=int64_t;using I=liboxns::LiBox<K,K>;
void check(bool ok){if(!ok)throw std::runtime_error("directory regression mismatch");}
int main(){
 const K n=4096,domain=n*4096;const std::string cfg="/tmp/directory-regression-"+std::to_string(getpid());struct Cleanup{std::string p;~Cleanup(){std::remove(p.c_str());}}cleanup{cfg};
 {std::ofstream f(cfg);for(K i=0;i<8;++i)f<<i*domain/8<<","<<(i+1)*domain/8<<",262144\n";}
 I idx(.5,.1,5);idx.loadConfigByFile(cfg);std::map<K,K>ref;std::vector<K>initial;
 auto put=[&](K k,K v){check(idx.insertKeyValue(k,v).status==liboxns::InsertStatus::SUCCESS);ref[k]=v;};
 auto get=[&](K k,K v){auto r=idx.searchKey(k);return r.status==liboxns::SearchStatus::SUCCESS&&r.value==v;};
 auto live=[&]{std::unordered_set<liboxns::Segment<K,K>*>active(idx.segments.begin(),idx.segments.end());check(!idx.redundantArray.empty());for(auto position:idx.redundantArray){check(position>=0&&size_t(position)<=idx.segments.size());check(active.count(idx.segments[std::min(size_t(position),idx.segments.size()-1)]));}};
 for(K i=0;i<n;++i){initial.push_back(i*4096);put(i*4096,(i*4096)^12345);}live();
 std::atomic<bool>stop{false};std::atomic<unsigned>errors{0};std::jthread reader([&]{std::mt19937_64 r(21);while(!stop.load()){K k=initial[r()%initial.size()];if(!get(k,k^12345))++errors;}});
 struct Stop{std::atomic<bool>&s;std::jthread&t;~Stop(){s.store(true);if(t.joinable())t.join();}}joiner{stop,reader};
 std::mt19937_64 rng(42);std::vector<K>added;while(added.size()<20000){K k=domain/2+K(rng()%40000);if(ref.count(k))continue;put(k,k^12345);added.push_back(k);if(added.size()%256==0)live();}
 stop.store(true);reader.join();check(!errors.load());live();
 for(size_t i=0;i<added.size();i+=3){check(idx.deleteKey(added[i]).found);ref.erase(added[i]);}
 for(size_t i=1;i<added.size();i+=3)put(added[i],-added[i]);
 for(auto[k,v]:ref)check(get(k,v));
 for(K i=0;i<1000;++i){K k=K(rng()%domain);auto r=idx.searchKey(k);check((r.status==liboxns::SearchStatus::SUCCESS)==bool(ref.count(k)));if(ref.count(k))check(r.value==ref.at(k));}
 const auto repaired=idx.redundantArray;const double scale=idx.a;idx.buildSearchIndex();check(scale==idx.a&&repaired==idx.redundantArray);live();for(auto[k,v]:ref)check(get(k,v));
}

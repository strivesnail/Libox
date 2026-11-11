#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include <omp.h>
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT=INT64_MAX;
using namespace liboxns;
using Index=LiBox<int64_t,int64_t>;
#else
#include "btreeolc.h"
using namespace ribtree;
using Index=BTree<int64_t,int64_t>;
#endif
using K=int64_t;
void require(bool ok,const std::string&why){if(!ok)throw std::runtime_error(why);}
void put(Index&idx,K k,K v){
#ifdef LIBOX
 require(idx.insertKeyValue(k,v).status==InsertStatus::SUCCESS,"index put");
#else
 idx.insert(k,v);
#endif
}
bool get(Index&idx,K k,K&v){
#ifdef LIBOX
 auto r=idx.searchKey(k);v=r.value;return r.status==SearchStatus::SUCCESS;
#else
 return idx.lookup(k,v);
#endif
}
std::map<K,K> collect(Index&idx,const std::map<K,K>&ref){std::map<K,K>out;auto add=[&](auto*s){for(auto[k,v]:s->getAllEntries())require(out.emplace(k,v).second,"duplicate physical key "+std::to_string(k));};
#ifdef LIBOX
 for(auto[k,expected]:ref){K v;require(get(idx,k,v)&&v==expected,"export/reference lookup mismatch");out[k]=v;}
#else
 std::function<void(NodeBase*)>visit=[&](auto*n){if(n->type==PageType::BTreeLeaf){auto*l=static_cast<BTreeLeaf<K,K>*>(n);for(unsigned i=0;i<l->count;++i)add(l->segments[i]);}else{auto*x=static_cast<BTreeInner<K>*>(n);for(unsigned i=0;i<=x->count;++i)visit(x->children[i]);}};visit(idx.root.load());
#endif
 return out;}
void index_case(const std::string&mode,int nt,unsigned seed){
#ifdef LIBOX
 Index idx(.5,.1,nt);
#else
 Index idx(nt,.5,.1);
#endif
 std::map<K,K>ref;K v;
 if(mode=="multiple_indexes"){
#ifdef LIBOX
  Index small(.5,.1,1),large(.5,.1,5);
#else
  Index small(1,.5,.1),large(5,.5,.1);
#endif
  omp_set_dynamic(0);omp_set_num_threads(5);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<4096;++i)put(large,i*512,i);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<4096;++i)put(small,i*512,-i-1);
  for(K i=0;i<4096;++i){require(get(large,i*512,v)&&v==i,"large index mismatch");require(get(small,i*512,v)&&v==-i-1,"small index mismatch");}
  return;
 }
 else if(mode=="invalid_threads"){
  for(int bad:{0,-1}){bool rejected=false;try{
#ifdef LIBOX
   Index invalid(.5,.1,bad);
#else
   Index invalid(bad,.5,.1);
#endif
  }catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid thread count accepted");}return;
 }
 else if(mode=="bulk_duplicates"||mode=="empty_bulk"){
  std::vector<std::pair<K,K>>pairs;if(mode=="bulk_duplicates")for(K i=0;i<4096;++i){K k=(i%128)*512;pairs.emplace_back(k,k^12345);ref[k]=k^12345;}
#ifdef LIBOX
  idx.bulk_load(pairs.data(),pairs.size());
#else
  idx.bulkLoad(pairs);
#endif
  if(mode=="empty_bulk")require(!get(idx,0,v),"empty bulk created key");
 }
 else if(mode=="duplicate_index"){for(int i=0;i<512;++i){put(idx,42,i);require(get(idx,42,v)&&v==i,"duplicate key did not update to latest value");}ref[42]=511;}
 else if(mode=="concurrent_duplicate"){
  omp_set_dynamic(0);omp_set_num_threads(nt);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<4096;++i)put(idx,(i%128)*512,((i+1LL)<<16)|(i%128));
  for(K k=0;k<128;++k){require(get(idx,k*512,v),"concurrent duplicate missing");require((v&65535)==k,"concurrent value/key mismatch");put(idx,k*512,-k-1);ref[k*512]=-k-1;}
 }else if(mode=="random_index"){
  std::mt19937_64 rng(seed);
  for(int step=0;step<20000;++step){K k=K(rng()%4096)-2048;K value=K(rng());if(rng()%4){put(idx,k,value);ref[k]=value;}else{bool exists=get(idx,k,v);require(exists==bool(ref.count(k))&&(!exists||v==ref.at(k)),"random lookup/reference mismatch at "+std::to_string(step));}if(step%1000==0)for(auto[k,vv]:ref)require(get(idx,k,v)&&v==vv,"reference mismatch at checkpoint");}
 }else if(mode=="empty_config"){
  std::string name="/tmp/index-empty-config-"+std::to_string(getpid())+".csv";std::ofstream(name).close();
#ifdef LIBOX
  idx.loadConfigByFile(name);
#else
  std::vector<std::pair<K,K>>empty;idx.bulkLoad(name,empty);
#endif
  std::remove(name.c_str());for(K i=0;i<4096;++i){put(idx,i*512,i);ref[i*512]=i;}
 }else throw std::runtime_error("bad index mode");
 for(auto[k,vv]:ref)require(get(idx,k,v)&&v==vv,"final reference mismatch");require(collect(idx,ref)==ref,"export/reference mismatch");
#ifdef LIBOX
 for(auto[k,vv]:ref){require(idx.deleteKey(k).found,"final deletion missing");require(!get(idx,k,v),"duplicate remained after deletion");}
#endif
}
void box_case(unsigned seed){ThreadIdManager::initialize(1);Box<K,K>b;std::map<K,K>ref;std::mt19937_64 rng(seed);
 for(int step=0;step<50000;++step){K k=K(rng()%256)-128,v=K(rng());unsigned op=rng()%4;if(op==0){bool exists=ref.count(k);auto r=b.insertKeyValue(k,v);if(exists||ref.size()<64){require(r.status==InsertStatus::SUCCESS,"insert/upsert failed");ref[k]=v;}else require(r.status==InsertStatus::FULL,"capacity exceeded");}else if(op==1){auto r=b.deleteKey(k);require(r.found==bool(ref.erase(k)),"delete mismatch");}else{auto r=b.searchKey(k);require((r.status==SearchStatus::SUCCESS)==bool(ref.count(k)),"box hit/miss mismatch");if(ref.count(k))require(r.value==ref[k],"box value mismatch");}
 if(step%50==0){require(b.getTotalCount()==ref.size(),"live count mismatch at "+std::to_string(step));std::map<K,K>actual;for(auto[k,vv]:b.getEntries())require(actual.emplace(k,vv).second,"export duplicates");require(actual==ref,"deleted/stale entries exported at "+std::to_string(step));std::vector<std::pair<K,K>>out;b.getEntriesInPlace(&out);require(std::map<K,K>(out.begin(),out.end())==ref&&out.size()==ref.size(),"in-place export mismatch");}
 }
}
void segment_case(unsigned seed){ThreadIdManager::initialize(1);auto make=[](){return std::make_unique<Segment<K,K>>(0,1000000,1000000,1);};auto s=make();std::map<K,K>ref;std::mt19937_64 rng(seed);const size_t cap=64*(1+overflowCapacity);
 for(int step=0;step<20000;++step){K k=rng()%512,v=K(rng());unsigned op=rng()%4;
  if(op<2){auto r=s->insertKeyValue(k,v);if(r.status==InsertStatus::FULL&&ref.size()<cap){auto saved=s->getAllEntries();require(std::map<K,K>(saved.begin(),saved.end())==ref&&saved.size()==ref.size(),"split export stale/duplicate");auto fresh=make();for(auto item:saved)require(fresh->insertKeyValue(item.first,item.second).status==InsertStatus::SUCCESS,"rebuild failed");s=std::move(fresh);r=s->insertKeyValue(k,v);}if(r.status==InsertStatus::SUCCESS)ref[k]=v;else require(r.status==InsertStatus::FULL&&!ref.count(k)&&ref.size()==cap,"unexpected segment insertion status");}
  else if(op==2){auto r=s->deleteKey(k);require(r.found==bool(ref.erase(k)),"segment delete mismatch");}
  else{auto r=s->searchKey(k);require((r.status==SearchStatus::SUCCESS)==bool(ref.count(k)),"segment lookup mismatch");if(ref.count(k))require(r.value==ref[k],"segment value mismatch");}
  if(step%100==0){auto saved=s->getAllEntries();require(std::map<K,K>(saved.begin(),saved.end())==ref&&saved.size()==ref.size(),"segment export mismatch");}
 }
}
int main(int argc,char**argv){std::string mode=argc>1?argv[1]:"box_reference";int nt=argc>2?std::stoi(argv[2]):1;unsigned seed=argc>3?std::stoul(argv[3]):42;
 if(mode=="box_reference")box_case(seed);
 else if(mode=="segment_reference")segment_case(seed);
 else if(mode=="segment_bounds"){ThreadIdManager::initialize(1);Segment<K,K>s(-100,100,10,1);for(K k:{K(-101),K(100),K(101),std::numeric_limits<K>::min(),std::numeric_limits<K>::max()}){require(s.searchKey(k).status!=SearchStatus::SUCCESS,"out-of-range hit");require(!s.deleteKey(k).found,"out-of-range delete");}}
 else if(mode=="split_admission"){ThreadIdManager::initialize(1);Segment<K,K>s(0,100,10,1);s.splitting_.store(true);require(s.deleteKey(0).status==DeleteStatus::SPLIT,"delete did not respect split admission");}
 else index_case(mode,nt,seed);
 std::cerr<<"PASS "<<mode<<" threads="<<nt<<" seed="<<seed<<"\n";
}

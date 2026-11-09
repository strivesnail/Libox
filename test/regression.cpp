#include <bits/stdc++.h>
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT = INT64_MAX;
using namespace liboxns;
using K=int64_t;
void require(bool c,const char* msg){if(!c)throw std::runtime_error(msg);}
int main(int argc,char**argv){
 std::string test=argc>1?argv[1]:"mixed";
 ThreadIdManager::initialize(1);
 if(test=="overflow"){
  Segment<K,K> seg(0,2048,1024,1);
  for(K k=0;k<320;k++)require(seg.insertKeyValue(k,k+77).status==InsertStatus::SUCCESS,"early full");
  require(seg.insertKeyValue(320,397).status==InsertStatus::FULL,"overflow escaped logical bucket");
  require(seg.searchKey(320).status==SearchStatus::NOT_FOUND,"overflow key should not exist");
  for(K k=0;k<320;k++)require(seg.searchKey(k).value==k+77,"overflow lost key");
 }else if(test=="deleted_export"){
  Box<K,K> b;
  for(K k=0;k<20;k++)b.insertKeyValue(k,k+77);
  for(K k=0;k<20;k+=2)require(b.deleteKey(k).found,"delete missing");
  auto entries=b.getEntries();require(entries.size()==10,"deleted entries exported");
  require(b.getTotalCount()==10,"invalid live count");
  for(auto [k,v]:entries)require(k%2&&v==k+77,"export resurrected deletion");
 }else if(test=="parallel"){
  const K n=8192;
  LiBox<K,K> idx(.5,.1,4);idx.loadConfigByFile(argv[2]);
  std::atomic<int> errors{0};
  omp_set_num_threads(4);
  #pragma omp parallel for
  for(K i=0;i<n;i++)if(idx.insertKeyValue(i*4096,i+77).status!=InsertStatus::SUCCESS)errors++;
  #pragma omp parallel for
  for(K i=0;i<n;i++){auto r=idx.searchKey(i*4096);if(r.status!=SearchStatus::SUCCESS||r.value!=i+77)errors++;}
  #pragma omp parallel for
  for(K i=0;i<n;i++)if(idx.insertKeyValue(i*4096,-i-77).status!=InsertStatus::SUCCESS)errors++;
  #pragma omp parallel for
  for(K i=0;i<n;i+=2)if(!idx.deleteKey(i*4096).found)errors++;
  #pragma omp parallel for
  for(K i=0;i<n;i++){auto r=idx.searchKey(i*4096);if(i%2?(r.status!=SearchStatus::SUCCESS||r.value!=-i-77):r.status==SearchStatus::SUCCESS)errors++;}
  require(errors==0,"parallel non-splitting operations failed");
 }else{
  const K n=8192,domain=n*4096;
  std::string conf=argv[2];
  LiBox<K,K> idx(.5,.1,1);idx.loadConfigByFile(conf);
  std::map<K,K> ref;
  auto put=[&](K k,K v){require(idx.insertKeyValue(k,v).status==InsertStatus::SUCCESS,"insert failed");ref[k]=v;};
  auto verify=[&](){for(auto [k,v]:ref){auto r=idx.searchKey(k);require(r.status==SearchStatus::SUCCESS&&r.value==v,"reference mismatch");}};
  for(K i=0;i<n;i++)put(i*4096,i*4096+77);
  for(K i=0;i<n;i+=3)put(i*4096,-i-19);
  std::vector<K> deleted;
  for(K i=0;i<n;i+=5){K k=i*4096;require(idx.deleteKey(k).found,"delete failed");ref.erase(k);deleted.push_back(k);}
  std::mt19937_64 rng(42);
  for(int i=0;i<16384;i++){K k=domain/2+rng()%(domain/100);put(k,k+91);}
  verify();
  for(K k:deleted)if(!ref.count(k))require(idx.searchKey(k).status!=SearchStatus::SUCCESS,"deleted key resurrected");
  for(K k:deleted)put(k,-k-77);
  verify();
  for(K k:{K(-1),domain,domain+100})require(idx.searchKey(k).status!=SearchStatus::SUCCESS,"outside key found");
  for(K i=0;i<1000;i++){K k=i*4096+1;if(!ref.count(k))require(idx.searchKey(k).status!=SearchStatus::SUCCESS,"missing key found");}
 }
 std::cerr<<"PASS "<<test<<"\n";
}

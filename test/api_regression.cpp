#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
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
void check(bool ok){if(!ok)throw std::runtime_error("API regression mismatch");}
std::unique_ptr<Index> make(){
#ifdef LIBOX
// GCC's uninstrumented libgomp reduction barrier is not visible to TSan.
#ifdef API_TSAN_SERIAL_BULK
return std::make_unique<Index>(.5,.1,1);
#else
return std::make_unique<Index>(.5,.1,5);
#endif
#else
return std::make_unique<Index>(5,.5,.1);
#endif
}
void put(Index&idx,K key,K value){
#ifdef LIBOX
check(idx.insertKeyValue(key,value).status==InsertStatus::SUCCESS);
#else
idx.insert(key,value);
#endif
}
bool get(Index&idx,K key,K&value){
#ifdef LIBOX
auto r=idx.searchKey(key);value=r.value;return r.status==SearchStatus::SUCCESS;
#else
return idx.lookup(key,value);
#endif
}
int main(){
 const std::string cfg="/tmp/index-api-regression-"+std::to_string(getpid());
 struct Cleanup{std::string path;~Cleanup(){std::remove(path.c_str());}}cleanup{cfg};
 auto write=[&](std::string text){std::ofstream f(cfg);f<<text;};
 auto idx=make();put(*idx,7,99);
 for(const std::string text:{"0,100\n","0,100,0\n","100,0,1\n","0,10,1\n20,30,1\n","0,20,1\n10,30,1\n","20,30,1\n0,20,1\n","0,100,-1\n","0,100,1garbage\n","0,100,1,\n","0,100,1,0,extra\n","nan,1,1\n"}){
  write(text);bool rejected=false;try{idx->loadConfigByFile(cfg);}catch(const std::exception&){rejected=true;}check(rejected);K v;check(get(*idx,7,v)&&v==99);
 }
 auto blank=make();put(*blank,7,99);write("\n \t\n");blank->loadConfigByFile(cfg);K old;check(!get(*blank,7,old));put(*blank,8,100);K value;check(get(*blank,8,value)&&value==100);
 ThreadIdManager::initialize(5);
 {Segment<double,K> f(0.,.5,1,5);check(f.getBoxCount()==1);check(f.insertKeyValue(.25,123).status==InsertStatus::SUCCESS);auto r=f.searchKey(.25);check(r.status==SearchStatus::SUCCESS&&r.value==123);}
 {auto parent=std::make_unique<Segment<K,K>>(0,100,10,5);auto alias=std::make_unique<Segment<K,K>>(50,100,10,5,parent->getBoxPtr(5),5,false,parent->getBoxStorageOwner());parent.reset();check(alias->insertKeyValue(75,456).status==InsertStatus::SUCCESS);check(alias->searchKey(75).value==456);}
#ifdef LIBOX
 for(bool repeated:{false,true}){auto generated=make();std::vector<K>keys;if(repeated)keys.assign(128,42);else for(K i=0;i<128;++i)keys.push_back((127-i)*10-640);generated->loadConfigFromData(keys);std::vector<std::pair<K,K>>pairs;for(K k:keys)pairs.emplace_back(k,k^12345);generated->bulk_load(pairs.data(),pairs.size());for(K k:keys)check(get(*generated,k,value)&&value==(k^12345));}
#else
 auto tree=make();std::string text;std::vector<K>keys;for(int i=0;i<40;++i){text+=std::to_string(i*100)+","+std::to_string((i+1)*100)+",100,0\n";keys.push_back(i*100+20);keys.push_back(i*100+3);keys.push_back(i*100+10);}write(text);tree->loadConfigByFile(cfg);for(K k:keys)put(*tree,k,k);std::sort(keys.begin(),keys.end());
 for(K start:{K(-1),K(0),K(4),K(100),K(1599),K(1600),K(3900),K(4000)})for(int count:{0,1,2,17,500}){std::vector<K>expected;for(K k:keys)if(k>=start&&expected.size()<size_t(count))expected.push_back(k);std::vector<K>out(count+1,-999);auto n=tree->scan(start,count,out.data());check(n==expected.size()&&std::equal(expected.begin(),expected.end(),out.begin())&&out[count]==-999);}
 put(*tree,3,999);K updated;check(tree->scan(0,1,&updated)==1&&updated==999);put(*tree,3,3);
 bool rejected=false;try{tree->scan(0,1,nullptr);}catch(const std::invalid_argument&){rejected=true;}check(rejected);check(tree->scan(0,0,nullptr)==0);
 auto growing=make();for(K k=0;k<1000;++k)put(*growing,k,k);std::atomic<bool>ready{false};std::jthread writer([&]{ready.store(true);for(K k=1000;k<5000;++k)put(*growing,k,k);});while(!ready.load())std::this_thread::yield();for(int i=0;i<10;++i){std::vector<K>out(5000);auto n=growing->scan(-1,5000,out.data());check(n>=1000&&n<=5000);for(size_t k=0;k<n;++k)check(out[k]==K(k));}writer.join();std::vector<K>out(5000);check(growing->scan(-1,5000,out.data())==5000);for(size_t k=0;k<out.size();++k)check(out[k]==K(k));
#endif
}

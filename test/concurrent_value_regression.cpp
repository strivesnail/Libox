#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT=INT64_MAX;
using namespace liboxns;
#else
#include "btreeolc.h"
using namespace ribtree;
#endif
int main(int argc,char**argv){int n=argc>1?std::stoi(argv[1]):200000;int nw=argc>2?std::stoi(argv[2]):1;ThreadIdManager::initialize(nw+1);auto fp=[](uint64_t b){b^=b>>33;b*=UINT64_C(0xff51afd7ed558ccd);b^=b>>33;uint8_t f=b>>56;return f?f:uint8_t(1);};int64_t a=9,b=10000;while(fp(b)!=fp(a))++b;Box<int64_t,int64_t>box;bool indexed=argc>3&&std::string(argv[3])=="index";
#ifdef LIBOX
 std::unique_ptr<LiBox<int64_t,int64_t>>idx;if(indexed)idx=std::make_unique<LiBox<int64_t,int64_t>>(.5,.1,nw+1);
#else
 std::unique_ptr<BTree<int64_t,int64_t>>idx;if(indexed)idx=std::make_unique<BTree<int64_t,int64_t>>(nw+1,.5,.1);
#endif
 auto put=[&](int64_t k,int64_t v){if(!indexed){box.insertKeyValue(k,v);return;}
#ifdef LIBOX
 idx->insertKeyValue(k,v);
#else
 idx->insert(k,v);
#endif
 };
 auto erase=[&](int64_t k){if(!indexed){box.deleteKey(k);return;}
#ifdef LIBOX
 idx->deleteKey(k);
#endif
 };
 auto get=[&](int64_t k){if(!indexed)return box.searchKey(k);
#ifdef LIBOX
 return idx->searchKey(k);
#else
 int64_t v{};bool ok=idx->lookup(k,v);return SearchResult<int64_t,int64_t>{ok?SearchStatus::SUCCESS:SearchStatus::NOT_FOUND,v};
#endif
 };put(a,100);std::atomic<bool>go{false};std::atomic<unsigned>errors{0};std::atomic<int>ready{0};std::vector<std::thread>writers;for(int t=0;t<nw;++t)writers.emplace_back([&]{++ready;while(!go.load()){}for(int i=0;i<n;++i){erase(a);put(b,200);erase(b);put(a,100);}});std::thread reader([&]{++ready;while(!go.load()){}for(int i=0;i<n*2;++i){auto x=get(a),y=get(b);if(x.status==SearchStatus::SUCCESS&&x.value!=100)++errors;if(y.status==SearchStatus::SUCCESS&&y.value!=200)++errors;}});while(ready.load()!=nw+1){}go=true;for(auto&w:writers)w.join();reader.join();auto x=get(a),y=get(b);if(x.status!=SearchStatus::SUCCESS||x.value!=100||(!indexed&&(y.status==SearchStatus::SUCCESS||box.getTotalCount()!=1)))++errors;std::cerr<<"value/key snapshot errors="<<errors.load()<<" writers="<<nw<<" iterations="<<n<<"\n";return errors?1:0;}

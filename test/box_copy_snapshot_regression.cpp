#include <bits/stdc++.h>
#include <sys/syscall.h>
#include <unistd.h>
#define private public
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT=INT64_MAX;
using namespace liboxns;
#else
#include "btreeolc.h"
using namespace ribtree;
#endif
#undef private
// Test-only scheduling: stop the copy at its normal wait for the source writer.
static thread_local bool copying=false;
static std::atomic<bool> paused{false},resume_copy{false};
extern "C" int sched_yield() noexcept {
    if(copying){paused.store(true,std::memory_order_release);while(!resume_copy.load(std::memory_order_acquire))__builtin_ia32_pause();}
    return static_cast<int>(syscall(SYS_sched_yield));
}
int main(){
    using B=Box<int64_t,int64_t>;
    B source;source.insertKeyValue(11,100);source.insertKeyValue(22,200);
    if(!source.try_acquire_write_lock())return 2;
    std::optional<B> result;
    std::thread reader([&]{copying=true;result.emplace(source);copying=false;});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!paused.load(std::memory_order_acquire)&&std::chrono::steady_clock::now()<deadline)__builtin_ia32_pause();
    if(!paused.load()){resume_copy=true;source.release_write_lock();reader.join();std::cerr<<"copy did not wait for source writer\n";return 3;}
    source.release_write_lock();
    // The source never contains key 11 together with value 201 for key 22.
    // Combining old validity/counts with the new payload is not a snapshot.
    source.deleteKey(11);source.insertKeyValue(22,201);
    resume_copy.store(true,std::memory_order_release);reader.join();
    const auto entries=result->getEntries();const auto a=result->searchKey(11),b=result->searchKey(22);
    bool ok=result->getTotalCount()==1&&entries==std::vector<std::pair<int64_t,int64_t>>{{22,201}}&&a.status!=SearchStatus::SUCCESS&&b.status==SearchStatus::SUCCESS&&b.value==201;
    std::cout<<"copy after writer handoff: count="<<result->getTotalCount()<<" entries="<<entries.size()<<" consistent="<<ok<<'\n';
    return ok?0:1;
}

#ifndef INDEX_ATOMIC_BOX_H
#define INDEX_ATOMIC_BOX_H
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

namespace index_concurrency {
// Version validation alone does not legalize racing ordinary C++ loads/stores.
// Cells remain compact, while atomic_ref makes every shared payload access legal.
template<class T> class AtomicCell {
    static_assert(std::is_trivially_copyable_v<T>);
    alignas(std::atomic_ref<T>::required_alignment) T value_{};
public:
    AtomicCell() = default;
    explicit AtomicCell(T value) : value_(value) {}
    AtomicCell(const AtomicCell& other) : value_(other.load()) {}
    T load() const { return std::atomic_ref<T>(const_cast<T&>(value_)).load(std::memory_order_acquire); }
    void store(T value) { std::atomic_ref<T>(value_).store(value,std::memory_order_release); }
    operator T() const { return load(); }
    AtomicCell& operator=(T value) { store(value); return *this; }
    AtomicCell& operator=(const AtomicCell& other) { return *this=other.load(); }
    // Box writers are already serialized: a second locked RMW is unnecessary.
    T operator++(int) { T old=load();store(old+1);return old; }
    T operator--(int) { T old=load();store(old-1);return old; }
};

class AtomicBitmap64 {
    std::atomic<uint64_t> bits_{0};
    struct Reference {
        AtomicBitmap64& owner; size_t slot;
        operator bool() const {return owner.test(slot);}
        Reference& operator=(bool value) {owner.set(slot,value);return *this;}
    };
public:
    AtomicBitmap64()=default;
    AtomicBitmap64(const AtomicBitmap64& other) : bits_(other.bits_.load(std::memory_order_acquire)) {}
    AtomicBitmap64& operator=(const AtomicBitmap64& other) {bits_.store(other.bits_.load(std::memory_order_acquire),std::memory_order_release);return *this;}
    bool test(size_t slot) const {return (bits_.load(std::memory_order_acquire)>>slot)&1;}
    void set(size_t slot,bool value) {auto old=bits_.load(std::memory_order_relaxed);uint64_t mask=uint64_t(1)<<slot;bits_.store(value?old|mask:old&~mask,std::memory_order_release);}
    Reference operator[](size_t slot) {return {*this,slot};}
    bool operator[](size_t slot) const {return test(slot);}
};

// A raw 512-bit SIMD load racing a fingerprint-byte store is a data race.
// Take eight atomic word snapshots, then compare the resulting local vector.
class alignas(64) AtomicFingerprints64 {
    std::array<std::atomic<uint64_t>,8> words_{};
    struct Reference {
        AtomicFingerprints64& owner;size_t slot;
        Reference& operator=(uint8_t value) {owner.set(slot,value);return *this;}
    };
public:
    AtomicFingerprints64()=default;
    AtomicFingerprints64(const AtomicFingerprints64& other) {*this=other;}
    AtomicFingerprints64& operator=(const AtomicFingerprints64& other) {for(size_t i=0;i<8;++i)words_[i].store(other.words_[i].load(std::memory_order_acquire),std::memory_order_release);return *this;}
    void set(size_t slot,uint8_t value) {auto& word=words_[slot/8];unsigned shift=unsigned(slot%8)*8;auto bits=word.load(std::memory_order_relaxed);word.store((bits&~(uint64_t(255)<<shift))|(uint64_t(value)<<shift),std::memory_order_release);}
    Reference operator[](size_t slot) {return {*this,slot};}
    const void* data() const {return words_.data();} // prefetch only
    __m512i loadVector() const {
        return _mm512_set_epi64(static_cast<long long>(words_[7].load(std::memory_order_acquire)),static_cast<long long>(words_[6].load(std::memory_order_acquire)),static_cast<long long>(words_[5].load(std::memory_order_acquire)),static_cast<long long>(words_[4].load(std::memory_order_acquire)),static_cast<long long>(words_[3].load(std::memory_order_acquire)),static_cast<long long>(words_[2].load(std::memory_order_acquire)),static_cast<long long>(words_[1].load(std::memory_order_acquire)),static_cast<long long>(words_[0].load(std::memory_order_acquire)));
    }
};
static_assert(sizeof(AtomicFingerprints64)==64);
}

#endif

#ifndef INDEX_ONLINE_RANGE_H
#define INDEX_ONLINE_RANGE_H
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace index_concurrency {
template<class Key> struct OnlineRange { Key lower, upper; size_t width; };
// Integer keys use half-open bounds. The largest representable key cannot
// itself be represented without a wider/sentinel upper-bound representation.
template<class Key> OnlineRange<Key> first_range(Key key) {
    if constexpr(!std::is_integral_v<Key>) {
        throw std::logic_error("online bootstrap requires integer keys; configure floating-key ranges first");
    } else {
    static_assert(sizeof(Key) <= 8);
    using Wide=__int128_t;
    const Wide maximum=std::numeric_limits<Key>::max();
    if (Wide(key)==maximum) throw std::out_of_range("maximum key has no representable exclusive upper bound");
    const Wide upper=std::min(Wide(key)+64,maximum);
    return {key,Key(upper),size_t(upper-Wide(key))};
    }
}
template<class Key> OnlineRange<Key> adjacent_range(Key key,Key lower,Key upper,bool before) {
    if constexpr(!std::is_integral_v<Key>) {
        throw std::logic_error("online range growth requires integer keys");
    } else {
    using Wide=__int128_t;
    const Wide minimum=std::numeric_limits<Key>::lowest(), maximum=std::numeric_limits<Key>::max();
    if(Wide(key)==maximum)throw std::out_of_range("maximum key has no representable exclusive upper bound");
    const Wide distance=before?Wide(lower)-Wide(key):Wide(key)-Wide(upper)+1;
    // Cap each interval so the segmentation algorithm's Key-sized window
    // remains representable. Extreme gaps may need more than one interval.
    const Wide width=std::min(maximum,std::max({Wide(64),distance,(Wide(upper)-Wide(lower))*2}));
    if(before){Wide start=std::max(minimum,Wide(lower)-width);return {Key(start),lower,size_t(Wide(lower)-start)};}
    Wide end=std::min(maximum,Wide(upper)+width);return {upper,Key(end),size_t(end-Wide(upper))};
    }
}
template<class Key> size_t range_width(Key lower,Key upper) {
    if constexpr(std::is_integral_v<Key>)return size_t(__int128_t(upper)-__int128_t(lower));
    else return size_t(upper-lower);
}
template<class Key> Key advance_bounded(Key start,Key width,Key limit) {
    if constexpr(std::is_integral_v<Key>)return Key(std::min(__int128_t(start)+__int128_t(width),__int128_t(limit)));
    else return std::min(start+width,limit);
}
}
#endif

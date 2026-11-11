#ifndef INDEX_CONFIG_H
#define INDEX_CONFIG_H
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
namespace index_concurrency {
inline std::string config_trim(std::string s) {
    auto first=s.find_first_not_of(" \t\r\n");
    return first==std::string::npos ? std::string{} : s.substr(first,s.find_last_not_of(" \t\r\n")-first+1);
}
template<class T>T config_number(const std::string& text) {
    const auto s=config_trim(text);size_t consumed=0;
    if(s.empty())throw std::invalid_argument("empty configuration field");
    if constexpr(std::is_floating_point_v<T>) {
        long double v=std::stold(s,&consumed);
        if(consumed!=s.size()||!std::isfinite(v)||v<std::numeric_limits<T>::lowest()||v>std::numeric_limits<T>::max())throw std::invalid_argument("invalid floating configuration field");
        return static_cast<T>(v);
    } else if constexpr(std::is_signed_v<T>) {
        auto v=std::stoll(s,&consumed);
        if(consumed!=s.size()||v<std::numeric_limits<T>::lowest()||v>std::numeric_limits<T>::max())throw std::invalid_argument("invalid signed configuration field");
        return static_cast<T>(v);
    } else {
        if(s.front()=='-')throw std::invalid_argument("negative unsigned configuration field");
        auto v=std::stoull(s,&consumed);
        if(consumed!=s.size()||v>std::numeric_limits<T>::max())throw std::invalid_argument("invalid unsigned configuration field");
        return static_cast<T>(v);
    }
}
template<class Key>struct ConfigRange {Key lower,upper;size_t width,count;};
template<class Key>std::vector<ConfigRange<Key>> read_index_config(const std::string& path) {
    std::ifstream file(path);if(!file)throw std::runtime_error("Failed to open config file: "+path);
    std::vector<ConfigRange<Key>> result;std::string line;size_t number=0;
    while(std::getline(file,line)) {
        ++number;if(config_trim(line).empty())continue;
        try {
            std::vector<std::string> fields;std::istringstream input(line);std::string field;
            while(std::getline(input,field,','))fields.push_back(field);
            if(!line.empty()&&line.back()==',')fields.emplace_back();
            if(fields.size()!=3&&fields.size()!=4)throw std::invalid_argument("expected three or four fields");
            ConfigRange<Key> r{config_number<Key>(fields[0]),config_number<Key>(fields[1]),config_number<size_t>(fields[2]),fields.size()==4?config_number<size_t>(fields[3]):0};
            if(!(r.lower<r.upper)||!r.width)throw std::invalid_argument("nonempty bounds and positive box width required");
            if(!result.empty()&&result.back().upper!=r.lower)throw std::invalid_argument("ranges must be ordered, contiguous and nonoverlapping");
            result.push_back(r);
        } catch(const std::exception& e) {throw std::invalid_argument("Invalid config line "+std::to_string(number)+": "+e.what());}
    }
    return result;
}
}
#endif

#ifndef KINEMA_CONFIG_H_
#define KINEMA_CONFIG_H_

// Container and string vocabulary. By default kinema is plain C++ on the
// standard library. With KINEMA_USE_BASE (the CMake option of the same name)
// the same names map onto equilibrium's base types instead, and kinema builds
// without the standard library or exceptions.
//
// The two sides are not API-compatible (base::Vector::at returns a pointer,
// base::StringRef::compare returns bool, base::Optional's move empties the
// source, ...), so library code only uses operations both provide with the
// same meaning, and the helpers below where they differ.

#include <stddef.h>
#include <stdint.h>

#ifdef KINEMA_USE_BASE
#include <base/containers/vector.h>
#include <base/memory/move.h>
#include <base/optional.h>
#include <base/strings/string_ref.h>
#include <base/strings/xstring.h>
#else
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#endif

namespace kinema {

#ifdef KINEMA_USE_BASE
// base's own scalar names (its u64 is unsigned long long, not uint64_t), so a
// host on base sees one type where kinema's and its own names meet.
using u8 = ::u8;
using u16 = ::u16;
using u32 = ::u32;
using u64 = ::u64;
using i16 = ::i16;
using f32 = ::f32;

template <typename T>
using Vector = base::Vector<T>;
template <typename T>
using Optional = base::Optional<T>;
using String = base::String;
using StringView = base::StringRef;
using base::move;
#else
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i16 = int16_t;
using f32 = float;

template <typename T>
using Vector = std::vector<T>;
template <typename T>
using Optional = std::optional<T>;
using String = std::string;
using StringView = std::string_view;
using std::move;
#endif

// std::min/max/clamp semantics on both sides (base::Min/Max/Clamp differ on
// NaN and signed zero): the first argument wins ties and unordered compares.
template <typename T>
constexpr const T& Min(const T& a, const T& b) {
  return b < a ? b : a;
}
template <typename T>
constexpr const T& Max(const T& a, const T& b) {
  return a < b ? b : a;
}
template <typename T>
constexpr const T& Clamp(const T& v, const T& lo, const T& hi) {
  return Min(Max(v, lo), hi);
}

// Stable insertion sort. Only builder-time code sorts, over inputs that are
// short and almost always already in order, where this is linear. Stable so
// the result is a function of the input alone: libstdc++'s std::sort is a plain
// insertion sort up to 16 elements, so this reproduces its order there exactly.
template <typename T, typename Less>
void StableSort(T* first, T* last, Less less) {
  if (first == last) return;
  for (T* i = first + 1; i < last; ++i) {
    T v = kinema::move(*i);
    T* j = i;
    for (; j > first && less(v, *(j - 1)); --j) *j = kinema::move(*(j - 1));
    *j = kinema::move(v);
  }
}

}  // namespace kinema

#endif  // KINEMA_CONFIG_H_

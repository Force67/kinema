// golden: emits FNV-1a 64 hashes of a set of deterministic curve-free clip
// blobs. It exists to guard the blob byte-layout across format revisions: the
// hashes below must not change when a new clip feature (ranged events, float
// curves, ...) is added but not used. Run it after a format change and diff the
// output against test/kinema_test.cc's baked kGolden* constants.
//
//   cmake --build build --target kinema_golden && ./build/tools/kinema_golden

#include <stdio.h>

#include "kinema/kinema.h"

namespace {

using namespace kinema;

u64 Fnv64(const Vector<u8>& bytes) {
  u64 h = 14695981039346656037ull;
  for (u8 b : bytes) {
    h ^= b;
    h *= 1099511628211ull;
  }
  return h;
}

// A v1 clip: constant + animated bone tracks, point events, root keys. No
// ranged events, no curves -> must stay byte-identical forever.
Vector<u8> BuildV1() {
  ClipBuilder b(6, 41, 30.0f);
  for (u32 f = 0; f < 41; ++f) {
    f32 t = static_cast<f32>(f) / 30.0f;
    for (u32 k = 0; k < 6; ++k) {
      Vec3 tr;
      Quat r;
      f32 s = 1.0f;
      if (k == 0) {  // constant track
        tr = Vec3{1.0f, 2.0f, 3.0f};
        r = Quat{0.0f, 0.0f, 0.1986693f, 0.9800666f};  // ~0.4 rad about z
      } else {
        f32 p = t * (0.5f + 0.3f * static_cast<f32>(k));
        tr = Vec3{4.0f * (p - static_cast<f32>(static_cast<int>(p))), 2.0f * static_cast<f32>(k),
                  1.5f};
        f32 half = 0.5f * (0.3f * p);
        r = Quat{0.0f, 0.0f, half, 1.0f - 0.5f * half * half};
        s = 1.0f + 0.1f * (p - static_cast<f32>(static_cast<int>(p)));
      }
      b.SetSample(f, k, tr, r, s);
    }
  }
  b.AddEvent("FootLeft", 0.4f);
  b.AddEvent("FootRight", 1.1f);
  b.AddRootKey(1.0f, Vec3{0.0f, 40.0f, 0.0f});
  return b.Build();
}

// A v2 clip: same as v1 plus a ranged event. Curve-free -> must stay
// byte-identical when curves are added to the format.
Vector<u8> BuildV2() {
  Vector<u8> unused = BuildV1();
  (void)unused;
  ClipBuilder b(6, 41, 30.0f);
  for (u32 f = 0; f < 41; ++f) {
    f32 t = static_cast<f32>(f) / 30.0f;
    for (u32 k = 0; k < 6; ++k) {
      Vec3 tr;
      Quat r;
      f32 s = 1.0f;
      if (k == 0) {
        tr = Vec3{1.0f, 2.0f, 3.0f};
        r = Quat{0.0f, 0.0f, 0.1986693f, 0.9800666f};
      } else {
        f32 p = t * (0.5f + 0.3f * static_cast<f32>(k));
        tr = Vec3{4.0f * (p - static_cast<f32>(static_cast<int>(p))), 2.0f * static_cast<f32>(k),
                  1.5f};
        f32 half = 0.5f * (0.3f * p);
        r = Quat{0.0f, 0.0f, half, 1.0f - 0.5f * half * half};
        s = 1.0f + 0.1f * (p - static_cast<f32>(static_cast<int>(p)));
      }
      b.SetSample(f, k, tr, r, s);
    }
  }
  b.AddEvent("FootLeft", 0.4f);
  b.AddEvent("FootRight", 1.1f);
  b.AddRangedEvent("Attack", 0.3f, 0.7f);
  b.AddRootKey(1.0f, Vec3{0.0f, 40.0f, 0.0f});
  return b.Build();
}

}  // namespace

int main() {
  Vector<u8> v1 = BuildV1();
  Vector<u8> v2 = BuildV2();
  printf("kGoldenV1 = 0x%016llxull;  // %zu bytes\n",
         static_cast<unsigned long long>(Fnv64(v1)), static_cast<size_t>(v1.size()));
  printf("kGoldenV2 = 0x%016llxull;  // %zu bytes\n",
         static_cast<unsigned long long>(Fnv64(v2)), static_cast<size_t>(v2.size()));
  return 0;
}

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
#ifndef X_HLS_ASSUME_SIM_H
#define X_HLS_ASSUME_SIM_H
#include <assert.h>

#ifndef __cplusplus

__attribute__((always_inline)) void hls_assume(int pred) { 
    assert(pred); 
}

#else

namespace hls {
    __attribute__((always_inline)) void assume(bool pred) { 
        assert(pred); 
    }
}  // namespace hls

#endif // __cplusplus
#endif  // X_HLS_ASSUME_SIM_H

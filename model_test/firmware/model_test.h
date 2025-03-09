#ifndef MODEL_TEST_H_
#define MODEL_TEST_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void model_test(
    input_t x_in[N_INPUT_1_1],
    result_t layer10_out[N_LAYER_8]
);


#endif

#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 57
#define N_LAYER_2 32
#define N_LAYER_2 32
#define N_LAYER_5 16
#define N_LAYER_5 16
#define N_LAYER_8 4
#define N_LAYER_8 4
#define N_OUTPUTS_2 4


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<12,4,AP_RND,AP_SAT,0> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<35,17> dense1_result_t;
typedef ap_fixed<16,6> dense1_weight_t;
typedef ap_fixed<16,6> dense1_bias_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> act1_param_t;
typedef ap_fixed<18,8> act1_table_t;
typedef ap_fixed<38,18> dense2_result_t;
typedef ap_fixed<16,6> dense2_weight_t;
typedef ap_fixed<16,6> dense2_bias_t;
typedef ap_uint<1> layer5_index;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> act2_param_t;
typedef ap_fixed<18,8> act2_table_t;
typedef ap_fixed<37,17> dense3_result_t;
typedef ap_fixed<16,6> dense3_weight_t;
typedef ap_fixed<16,6> dense3_bias_t;
typedef ap_uint<1> layer8_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<16,6> act3_param_t;
typedef ap_fixed<18,8> act3_table_t;


#endif

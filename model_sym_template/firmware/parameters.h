#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "hls_math.h"
#include "nnet_utils/nnet_math.h"

// hls-fpga-machine-learning insert weights


// hls-fpga-machine-learning insert layer-config
// expr1
nnet::lookup_table<result_t, 256, hls::tanh> tanh_lut(-32, 32);
nnet::lookup_table<result_t, 256, hls::log> log_lut(0, 128);
nnet::lookup_table<result_t, 256, hls::abs> abs_lut(-32, 32);


#endif

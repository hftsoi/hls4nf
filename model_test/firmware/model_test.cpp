#include <iostream>

#include "model_test.h"
#include "parameters.h"


void model_test(
    input_t x_in[N_INPUT_1_1],
    result_t layer10_out[N_LAYER_8]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x_in complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x_in,layer10_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<dense1_weight_t, 1824>(w2, "w2.txt");
        nnet::load_weights_from_txt<dense1_bias_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<dense2_weight_t, 512>(w5, "w5.txt");
        nnet::load_weights_from_txt<dense2_bias_t, 16>(b5, "b5.txt");
        nnet::load_weights_from_txt<dense3_weight_t, 64>(w8, "w8.txt");
        nnet::load_weights_from_txt<dense3_bias_t, 4>(b8, "b8.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    dense1_result_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, dense1_result_t, config2>(x_in, layer2_out, w2, b2); // dense1

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::leaky_relu<dense1_result_t, act1_param_t, layer4_t, LeakyReLU_config4>(layer2_out, 0.10000000149011612, layer4_out); // act1

    dense2_result_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense<layer4_t, dense2_result_t, config5>(layer4_out, layer5_out, w5, b5); // dense2

    layer7_t layer7_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::leaky_relu<dense2_result_t, act2_param_t, layer7_t, LeakyReLU_config7>(layer5_out, 0.10000000149011612, layer7_out); // act2

    dense3_result_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, dense3_result_t, config8>(layer7_out, layer8_out, w8, b8); // dense3

    nnet::leaky_relu<dense3_result_t, act3_param_t, result_t, LeakyReLU_config10>(layer8_out, 0.10000000149011612, layer10_out); // act3

}


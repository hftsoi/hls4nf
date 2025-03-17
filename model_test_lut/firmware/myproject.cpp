#include <iostream>

#include "myproject.h"
#include "parameters.h"


#include "hls_math.h"
nnet::lookup_table<result_t, 256, hls::tanh<16,6>> tanh_lut(-4, 4);
nnet::lookup_table<result_t, 512, hls::log<16,6>> log_lut(0.125, 64);

template <class data_T, class res_T, class weight_T, int N, class Op_tanh, class Op_log>
void flow_planar(const data_T x[N],
                 const weight_T flow_w[N],
                 const weight_T flow_u[N],
                 const weight_T &flow_b,
                 const Op_tanh &op_tanh,
                 const Op_log &op_log,
                 res_T z[N],
                 res_T &log_det_total) {
FlowTransform:
    // z = x + u * h(w^T x + b)
    res_T linear_term = 0;
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL
        linear_term += flow_w[i] * x[i];
    }
    res_T h = op_tanh(linear_term + flow_b);
    
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL
        z[i] = x[i] + flow_u[i] * h;
    }

LogDeterminant:
    // log|det| = log|1 + u^T h'(w^T x + b) * w|
    res_T det = 0;
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL
        det += flow_u[i] * flow_w[i];
    }
    // h' = 1 - h^2 for h = tanh
    det = 1 + det * (1 - h * h);
    
    if (det > 0) {
        log_det_total += op_log(det);
    } else {
        log_det_total += op_log(-det);
    }
}


void myproject(
    input_t x[N_INPUT_1_1],
    result_t y[N_OUTPUTS_2]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=y complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,y 
    #pragma HLS PIPELINE
    //#pragma HLS DATAFLOW

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
    nnet::dense<input_t, dense1_result_t, config2>(x, layer2_out, w2, b2); // dense1

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::leaky_relu<dense1_result_t, act1_param_t, layer4_t, LeakyReLU_config4>(layer2_out, 0.1, layer4_out); // act1

    dense2_result_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense<layer4_t, dense2_result_t, config5>(layer4_out, layer5_out, w5, b5); // dense2

    layer7_t layer7_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::leaky_relu<dense2_result_t, act2_param_t, layer7_t, LeakyReLU_config7>(layer5_out, 0.1, layer7_out); // act2

    dense3_result_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, dense3_result_t, config8>(layer7_out, layer8_out, w8, b8); // dense3
    
    result_t layer10_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::leaky_relu<dense3_result_t, act3_param_t, result_t, LeakyReLU_config10>(layer8_out, 0.1, layer10_out); // act3
    
    ap_fixed<16,6> flow0_w[4] = {-12.59094, -11.641701, 11.871289, 12.841861};
    ap_fixed<16,6> flow0_u[4] = {-6.976981, -6.9543624, 6.4214606, 7.1096206};
    ap_fixed<16,6> flow0_b[1] = {0.01299851};
    #pragma HLS ARRAY_PARTITION variable=flow0_w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=flow0_u complete dim=0
    #pragma HLS ARRAY_PARTITION variable=flow0_b complete dim=0

    ap_fixed<16,6> flow1_w[4] = {6.817665, -7.156333, 8.390457, -7.8095684};
    ap_fixed<16,6> flow1_u[4] = {7.0930486, -7.359327, 7.948529, -8.14267};
    ap_fixed<16,6> flow1_b[1] = {-0.02948051};
    #pragma HLS ARRAY_PARTITION variable=flow1_w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=flow1_u complete dim=0
    #pragma HLS ARRAY_PARTITION variable=flow1_b complete dim=0

    ap_fixed<16,6> flow2_w[4] = {6.584381, 6.518894, 6.8391137, 6.5339704};
    ap_fixed<16,6> flow2_u[4] = {23.484638, 24.631702, 24.786816, 25.242905};
    ap_fixed<16,6> flow2_b[1] = {1.9243195};
    #pragma HLS ARRAY_PARTITION variable=flow2_w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=flow2_u complete dim=0
    #pragma HLS ARRAY_PARTITION variable=flow2_b complete dim=0


    result_t log_det_total = 0;

    result_t flow0_out[4];
    #pragma HLS ARRAY_PARTITION variable=flow0_out complete dim=0
    flow_planar<result_t, result_t, ap_fixed<16,6>, 4>(layer10_out, flow0_w, flow0_u, flow0_b, tanh_lut, log_lut, flow0_out, log_det_total);

    result_t flow1_out[4];
    #pragma HLS ARRAY_PARTITION variable=flow1_out complete dim=0
    flow_planar<result_t, result_t, ap_fixed<16,6>, 4>(flow0_out, flow1_w, flow1_u, flow1_b, tanh_lut, log_lut, flow1_out, log_det_total);

    flow_planar<result_t, result_t, ap_fixed<16,6>, 4>(flow1_out, flow2_w, flow2_u, flow2_b, tanh_lut, log_lut, y, log_det_total);
    
    //planar_flow(layer10_out, y, log_det_total, flow0_w, flow0_u, flow0_b);

    //std::cout << "log_det " << log_det_total[0] << std::endl;

}


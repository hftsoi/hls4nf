import hls4ml

expr = ['x0*x0*x0+tanh_lut(x0)+log_lut(x0)+abs_lut(x0)']

function_definitions = [
    'tanh_lut(x) = math_lut(tanh, x, N=256, range_start=-32, range_end=32)',
    'log_lut(x) = math_lut(log, x, N=256, range_start=0, range_end=128)',
    'abs_lut(x) = math_lut(abs, x, N=256, range_start=-32, range_end=32)',
]

#from hls4ml.utils.symbolic_utils import init_pysr_lut_functions
#init_pysr_lut_functions(init_defaults=True, function_definitions=function_definitions)

lut_functions = {
    'tanh_lut': {'math_func': 'tanh', 'range_start': -32, 'range_end': 32, 'table_size': 256},
    'log_lut': {'math_func': 'log', 'range_start': 0, 'range_end': 128, 'table_size': 256},
    'abs_lut': {'math_func': 'abs', 'range_start': -32, 'range_end': 32, 'table_size': 256},
}


hls_model_lut = hls4ml.converters.convert_from_symbolic_expression(
    expr,
    n_symbols=1,
    output_dir='model_sym_template',
    precision='ap_fixed<16,6>',
    part='xcvu13p-flga2577-2-e',
    lut_functions=lut_functions,
    hls_include_path='/tools/Xilinx/Vitis_HLS/2023.1/include',
    hls_libs_path='/tools/Xilinx/Vitis_HLS/2023.1/lnx64'
)
hls_model_lut.write()
#hls_model_lut.compile()


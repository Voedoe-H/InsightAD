import onnx
from onnx import shape_inference
import hls4ml
import hls4ml.converters.onnx_to_hls as onnx_to_hls

def patched_get_global_input_shape(graph, inp):
    for x in graph.input:
        if x.name == inp:
            return [dim.dim_value for dim in x.type.tensor_type.shape.dim]
    for x in graph.value_info:
        if x.name == inp:
            return [dim.dim_value for dim in x.type.tensor_type.shape.dim]
    for x in graph.output:
        if x.name == inp:
            return [dim.dim_value for dim in x.type.tensor_type.shape.dim]
    raise ValueError(f"Could not find shape information for tensor '{inp}'")

def patched_get_input_shape(graph, node):
    input_shapes = []
    for inp in node.input:
        try:
            dims = patched_get_global_input_shape(graph, inp)
            input_shapes.append(dims)
        except Exception as e:
            # fallback or raise more informative error
            raise RuntimeError(f"Shape not found for input '{inp}' in node '{node.name}': {e}")
    return input_shapes

# Patch both functions
onnx_to_hls.get_global_input_shape = patched_get_global_input_shape
onnx_to_hls.get_input_shape = patched_get_input_shape

# Load inferred model
onnx_model = onnx.load('pytorchmodel_v2_inferred.onnx')

config = hls4ml.utils.config_from_onnx_model(
    onnx_model,
    granularity='name',
    backend='Vivado'
)

def d():
    model_path = 'pytorchmodel_v2.onnx'
    onnx_model = onnx.load(model_path)

    # Run shape inference
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # Save the model with inferred shapes
    onnx.save(inferred_model, 'pytorchmodel_v2_inferred.onnx')



def synth():
    model_path = 'pytorchmodel_v2.onnx'  
    onnx_model = onnx.load(model_path)

    config = hls4ml.utils.config_from_onnx_model(
        onnx_model,
        granularity='name',
        backend='Vivado'  # or 'Vitis' or 'Quartus' depending on your target
    )

    pprint(config)

    for layer in config['LayerName']:
        config['LayerName'][layer]['Precision'] = 'ap_fixed<16,6>'
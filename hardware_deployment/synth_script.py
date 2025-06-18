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

def d():
    model_path = 'pytorchmodel_v2.onnx'
    onnx_model = onnx.load(model_path)

    # Run shape inference
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # Save the model with inferred shapes
    onnx.save(inferred_model, 'pytorchmodel_v2_inferred.onnx')



def synth():
    model = onnx.load('pytorchmodel_v2_inferred.onnx')

    # Rename input
    for i in range(len(model.graph.input)):
        model.graph.input[i].name = 'inp'

    # Rename output
    for i in range(len(model.graph.output)):
        model.graph.output[i].name = 'out'

    # Also rename initializers and nodes to match if needed
    for node in model.graph.node:
        node.input[:] = ['inp' if x == 'input' else x for x in node.input]
        node.output[:] = ['out' if x == 'output' else x for x in node.output]

    # Save the modified model
    onnx.save(model, 'pytorchmodel_v2_hls4ml_ready.onnx')


def fix_batch_size(model_path, output_path, input_name='inp', output_name='out'):
    model = onnx.load(model_path)

    # Save original names
    old_input_name = model.graph.input[0].name
    old_output_name = model.graph.output[0].name

    # Fix batch size in input/output/value_info
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        shape = value.type.tensor_type.shape
        if shape.dim and shape.dim[0].dim_value == 0:
            shape.dim[0].dim_value = 1

    # Rename input in graph.input
    if input_name and model.graph.input:
        print(f"Renaming input: {old_input_name} -> {input_name}")
        model.graph.input[0].name = input_name

    # Rename output in graph.output
    if output_name and model.graph.output:
        print(f"Renaming output: {old_output_name} -> {output_name}")
        model.graph.output[0].name = output_name

    # Replace all node input/output references
    for node in model.graph.node:
        node.input[:] = [input_name if x == old_input_name else x for x in node.input]
        node.output[:] = [output_name if x == old_output_name else x for x in node.output]

    # Same for value_info
    for v in model.graph.value_info:
        if v.name == old_input_name:
            v.name = input_name
        elif v.name == old_output_name:
            v.name = output_name

    onnx.save(model, output_path)
    print(f"Saved patched model to: {output_path}")

# Use like this
fix_batch_size(
    model_path='pytorchmodel_v2_inferred.onnx',
    output_path='pytorchmodel_v2_hls4ml_ready.onnx',
    input_name='inp',
    output_name='out'
)


#synth()
onnx_model = onnx.load('pytorchmodel_v2_hls4ml_ready.onnx')

print("Model inputs:")
for i in onnx_model.graph.input:
    print(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])

print("Model outputs:")
for o in onnx_model.graph.output:
    print(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])

print("Value info:")
for v in onnx_model.graph.value_info:
    print(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim])

onnx_model = onnx.load('pytorchmodel_v2_hls4ml_ready.onnx')
config = hls4ml.utils.config_from_onnx_model(onnx_model, granularity='name', backend='Vivado')
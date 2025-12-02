import onnx
m = onnx.load("best.onnx")
print("Inputs:", [i.name for i in m.graph.input])
for i in m.graph.input:
    dims = [d.dim_value if d.dim_value > 0 else 'dyn' for d in i.type.tensor_type.shape.dim]
    print(f"{i.name}: {dims}")

import onnx

def print_bias_and_weight_params(onnx_file_path):
    # 加载 ONNX 模型
    model = onnx.load(onnx_file_path)
    graph = model.graph

    # 遍历初始化参数，寻找 bias 和 weight
    for initializer in graph.initializer:
        name = initializer.name
        if "bias" in name.lower() or "weight" in name.lower():
            print(f"Parameter Name: {name}")
            print(f"Shape: {initializer.dims}")
            print(f"Values: {onnx.numpy_helper.to_array(initializer)}\n")

# 使用时替换为你的 .onnx 文件路径
onnx_file_path = "gonet.onnx"
print_bias_and_weight_params(onnx_file_path)

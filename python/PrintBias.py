import onnx
import numpy as np

# 加载 ONNX 模型
model_path = "gonet.onnx"  # 替换为你的 ONNX 文件路径
model = onnx.load(model_path)

# 获取权重参数
weights = onnx.numpy_helper.to_array

# 打印所有 bias 参数
for initializer in model.graph.initializer:
    if "bias" in initializer.name.lower():  # 检查名字是否包含 "bias"
        print(f"Parameter name: {initializer.name}")
        print(f"Bias values: {weights(initializer)}\n")

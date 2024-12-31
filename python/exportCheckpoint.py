import torch
import torch.nn as nn
import torch.onnx



# Define the neural network
class GoNet(nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()

        # 第一层卷积：输入通道 4，输出通道 192，卷积核大小 5x5，步幅 1，填充 2（保持尺寸）
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=192, kernel_size=5, stride=1, padding=2)
        # 后续 11 层卷积：输入通道 192，输出通道 192，卷积核大小 3x3，步幅 1，填充 1（保持尺寸）
        self.hidden_convs = nn.Sequential(
            *[nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1) for _ in range(11)]
        )
        # 输出层：1x1 卷积，输入通道 192，输出通道 1（对应棋盘每个位置的概率）
        self.output_conv = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, stride=1, padding=0)
        # 激活函数
        self.relu = nn.ReLU()
        # Softmax 层：在 2D 平面上按每个位置归一化概率
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 第一层卷积 + ReLU
        x = self.relu(self.conv1(x))
        # 11 层隐藏卷积 + ReLU
        for conv in self.hidden_convs:
            x = self.relu(conv(x))
        # 最后一层卷积
        x = self.output_conv(x)
        # 输出概率分布（flatten 为 [batch_size, 361]，然后应用 softmax）
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 361]
        # x = self.softmax(x)  # Apply softmax to get probabilities
        return x

# 定义一个加载函数
def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]

# 初始化模型和优化器
model = GoNet()
optimizer = torch.optim.SGD(model.parameters())

# 加载检查点
checkpoint_path = "checkpoint.pth"
epoch = load_checkpoint(checkpoint_path, model, optimizer)

def convert_to_onnx(model, onnx_path="gonet.onnx"):
    model.eval()  # 切换到评估模式

    # 定义一个 dummy 输入（假设输入是 [batch_size, 4, 19, 19]）
    dummy_input = torch.randn(1, 4, 19, 19)

    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,  # ONNX opset 版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=["input"],      # 输入名称
        output_names=["output"],    # 输出名称
        dynamic_axes={              # 支持动态 batch size
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"模型已保存为 {onnx_path}")

# 转换模型
convert_to_onnx(model)

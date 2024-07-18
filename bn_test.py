import torch
import numpy as np
from model import MLP

# 设置随机种子以确保结果的可复现性
torch.manual_seed(42)

# 初始化模型
model = MLP(784, hidden_features=128)  # 假设隐藏层大小为128

# 读取线性层的输出数据
linear_output = np.loadtxt('linear_output.txt')
linear_layer_output = torch.from_numpy(linear_output).float()

# 提取BN层的参数
bn_weight = model.fc1_bn.weight.data.numpy()
bn_bias = model.fc1_bn.bias.data.numpy()
running_mean = model.fc1_bn.running_mean.numpy()
running_var = model.fc1_bn.running_var.numpy()

# 将参数保存到文本文件
np.savetxt('bn_weight.txt', bn_weight)
np.savetxt('bn_bias.txt', bn_bias)
np.savetxt('bn_running_mean.txt', running_mean)
np.savetxt('bn_running_var.txt', running_var)

# BN层的前向传播
def bn_forward(input_data, weight, bias, running_mean, running_var, eps=1e-5):
    std = (running_var + eps) ** 0.5
    normalized = (input_data - running_mean) / std
    # 将weight和bias转换为Tensor类型
    weight_tensor = torch.from_numpy(weight)
    bias_tensor = torch.from_numpy(bias)
    output = weight_tensor * normalized + bias_tensor
    return output

# 使用BN层的前向传播
bn_output = bn_forward(linear_layer_output, bn_weight, bn_bias, running_mean, running_var)

# 保存BN层的输出到文件
np.savetxt('bn_output.txt', bn_output.numpy())

print("BN层的输出和参数已保存到相应的文本文件中。")
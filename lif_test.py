import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import MultiStepLIFNode  # 假设MultiStepLIFNode类在model模块中实现

# 设置随机种子以确保结果的可复现性
torch.manual_seed(42)

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载MNIST数据集
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 初始化LIF神经元层
lif_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

# 获取数据集中第一张图片
image, _ = dataset[0]

# 预处理图像数据
preprocessed_image = image.view(-1).float()

# 定义时间步长
time_steps = 100

# 构造一个序列化的输入，即使对于单个时间步，也需要将输入数据包装成序列的形式
input_sequence = preprocessed_image.unsqueeze(0).repeat(time_steps, 1)

# LIF神经元层的前向传播
lif_output = lif_neuron(input_sequence)

# 打印LIF神经元层的输出形状
print("LIF Neuron Output Shape:", lif_output.shape)
print("LIF Neuron Output for Time Step 0:", lif_output[0].numpy().flatten())

# 保存LIF神经元层的输出到文件
import numpy as np
np.savetxt('lif_output.txt', lif_output.numpy())

# 保存内部状态
# 注意：膜电位序列保存为numpy数组
membrane_potentials = lif_neuron.v_seq.numpy()

# 阈值是固定的，只需保存一次
threshold = lif_neuron.v_threshold

# 保存内部状态
np.savetxt('membrane_potentials.txt', membrane_potentials)
np.savetxt('threshold.txt', [threshold])
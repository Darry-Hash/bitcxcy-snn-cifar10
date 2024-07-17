import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import MLP

# 设置随机种子
torch.manual_seed(42)

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载MNIST数据集
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 初始化模型
model = MLP(784, hidden_features=128)  # 假设隐藏层大小为128，简化处理

# 获取数据集中第一张图片
image, _ = dataset[0]

# 打印原始图像数据经过预处理后的形状和内容
preprocessed_image = image.view(-1).float()
print("Preprocessed Image Data Shape:", preprocessed_image.shape)
print("Preprocessed Image Data:", preprocessed_image.numpy().flatten())  # 打印出全部预处理后的图像数据

# 假设只测试fc1_linear层
fc1_weights = model.fc1_linear.weight.data.numpy()
fc1_bias = model.fc1_linear.bias.data.numpy()


def linear_forward(input_data, weights, bias):
    # 确保输入为二维
    input_data = input_data.unsqueeze(0) if input_data.dim() == 1 else input_data
    output = torch.mm(input_data, torch.tensor(weights).t())
    if bias is not None:
        output += torch.tensor(bias).unsqueeze(0)
    return output


mnist_image_data = preprocessed_image  # 使用预处理后的图像数据
output_python = linear_forward(mnist_image_data, fc1_weights, fc1_bias)

print("\nPython Output After Linear Layer:", output_python.numpy().flatten())

# 保存权重和偏置到文件中
with open('fc1_weights.txt', 'w') as f:
    for weight in fc1_weights.flatten():
        f.write(f'{weight},')

with open('fc1_bias.txt', 'w') as f:
    for b in fc1_bias:
        f.write(f'{b},')

print("Weights and biases saved to 'fc1_weights.txt' and 'fc1_bias.txt'.")

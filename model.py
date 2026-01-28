import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CnnNet(nn.Module):
    def __init__(self, classes=10):
        super(CnnNet, self).__init__()
        self.classes = classes
        # 第一层卷积：输入bs*1*28*28 → 输出bs*16*14*14
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化后尺寸减半：28→14
        )
        # 第二层卷积：输入bs*16*14*14 → 输出bs*32*7*7
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化后尺寸减半：14→7
        )
        # 第三层卷积：输入bs*32*7*7 → 输出bs*64*3*3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化后尺寸：7→3（向下取整）
        )
        # 自适应平均池化：将任意尺寸映射为1*1，兼容尺寸变化
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层：64维特征→10分类
        self.fc = nn.Linear(64, self.classes)

    def forward(self, x):
        # 输入为4维特征图[bs,1,28,28]，直接送入卷积层（无需展平）
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.advpool(x)
        # 展平：[bs,64,1,1] → [bs,64]，保留batch_size维度
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST(root='mnist_data/', train=is_train, download=True, transform=to_tensor)
    return DataLoader(data_set, batch_size=128, shuffle=True, pin_memory=True)


def evaluate(test_data, net):
    net.eval()  # 将模型设置为评估模式
    n_correct = 0
    n_total = 0
    # n_correct：用于记录模型预测正确的样本数量。
    # n_total：用于记录测试样本的总数量。
    with torch.no_grad():  # 禁用梯度计算。在评估模型时，我们不需要计算梯度，因为不会进行反向传播。使用 torch.no_grad() 可以节省内存并提高计算效率。
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)  # 移动数据到设备
            outputs = net(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = CnnNet()
    net.to(device)  # 将模型移动到选择的设备

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epochs = 15

    for epoch in range(epochs):
        net.train()
        train_losses = []  # 记录每批次训练损失
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            net.zero_grad()  # 清空模型梯度
            outputs = net(x)
            loss = torch.nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())  # 收集损失

        # 计算训练日志
        avg_train_loss = torch.tensor(train_losses).mean().item()
        train_acc = evaluate(train_data, net)
        test_acc = evaluate(test_data, net)
        # 打印详细日志
        print(
            f"Epoch: {epoch + 1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    plt.figure(figsize=(10, 5))
    for n, (x, _) in enumerate(test_data):
        if n > 3:
            break
        x = x.to(device)
        outputs = net(x)
        prediction = torch.argmax(outputs, dim=1)[0]
        plt.subplot(2, 2, n + 1)
        plt.imshow(x[0].view(28, 28).cpu().numpy(), cmap='gray')  # 确保在绘图前数据在CPU上
        plt.title("Prediction: " + str(prediction.item()))
        plt.axis('off')
    plt.show()

    torch.save(net, "model.pth")


if __name__ == '__main__':
    main()


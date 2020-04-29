from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
plt.ion() # 交互模式

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_data = datasets.MNIST(root='.',
               train=True,
               download=True,
               transform= transform)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = 64,
    shuffle = True,
    num_workers = 4)

test_data = datasets.MNIST(root='.',
                   train=False,
                   transform= transform)
# 测试数据集
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size = 64,shuffle=True,num_workers=4)

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # 空间变换器定位 -- 网络
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = 7),
            nn.MaxPool2d(2, stride =2),
            nn.ReLU(True),

            nn.Conv2d(8,10, kernel_size = 5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # 3*2 affine 矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(10*3*3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*2)
        )

        # 使用身份转换初始化权重 / 偏差
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0],
                                                    dtype=torch.float))

    # 空间变换器网络转发功能
    def stn(self, x):
        xs = self.localization(x)#  1--10

        xs = xs.view(-1, 10*3*3)

        theta = self.fc_loc(xs) # 90 -- 6
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # 执行一般的前进传递
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training =self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)
# train model
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx%500 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100. *batch_idx / len(train_loader),
                loss.item()
            ))

# 测试函数
def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data , target = data.to(device), target.to(device)
            output = model(data)

            # 累加批量损失
            test_loss += F.nll_loss(output, target, size_average=False).item()

            # 获取最大对数概率的索引
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f} %))\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

# 可视化 STN 结果
def convert_image_np(inp):

    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    inp = std*inp +mean

    inp = np.clip(inp, 0, 1)
    return inp

# STN 可视化一批输入图像和相应变换批次

def visualize_stn():
    with torch.no_grad():
        data = next(iter(test_loader))[0].to(device)
        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by_side
        f, axarr= plt.subplots(1,2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title("Dataset Images")

        axarr[1].imshow(out_grid)
        axarr[1].set_title("Transformed Images")


for epoch in range(1, 20+1):
    train(epoch)
    test()

plt.ioff()
plt.imshow()




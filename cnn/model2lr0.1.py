import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# 使用FashionMNIST数据，准备训练数据集
train_data = FashionMNIST(
    root='../data/FashionMNIST',  # 数据路径
    train=True,  # 只使用训练数据集
    transform=transforms.ToTensor(),
    download=False
)
# 定义一个数据加载器
train_loader = Data.DataLoader(
    dataset=train_data,  # 使用的数据集
    batch_size=64,  # 批处理大小
    shuffle=False,  # 每次迭代前不打乱数据
    num_workers=0,  # 使用两个进程
)
# 计算有多少个batch
# print("train_loader的batch数量为：",len(train_loader))
# 获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
    print(b_x.size(0))
# print(b_y.shape)
print(b_x.shape)
# 可视化一个batch图像
# batch_x = b_x.squeeze().numpy()
# batch_y = b_y.numpy()
class_label = train_data.classes
class_label[0] = "T-shirt"
# plt.figure(figsize=(12, 5))
# for ii in np.arange(len(batch_y)):
#     plt.subplot(4, 16, ii + 1)
#     plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
#     plt.title(class_label[batch_y[ii]], size=9)
#     plt.axis("off")
#     plt.subplots_adjust(wspace=0.05)
# 对测试集进行处理
test_data = FashionMNIST(
    root='../data/FashionMNIST',  # 数据路径
    train=False,  # 不使用训练数据集
    download=False
)
# 为数据添加一个通道维度，并且取值范围放缩到0-1之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets  # 测试集的标签
# print("test_data_x.shape:", test_data_x.shape)
# print("test_data_y.shape:", test_data_y.shape)


# 搭建卷积神经网络
class MyConvnet(nn.Module):
    def __init__(self):
        super(MyConvnet, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入的feature map，输入通道数
                out_channels=16,  # 输出的feature map，输出通道数
                stride=1,  # 卷积核步长
                kernel_size=3,  # 卷积核尺寸
                padding=1,  # 进行填充
            ),  # 卷积后：（1*28*28） ->(16*28*28)
            nn.ReLU(),  # 激活函数
            nn.AvgPool2d(
                kernel_size=2,  # 平均值池化层，使用2*2
            ),  # 池化后：（16*28*28） ->(16*14*14)
        )
        # 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0),  # 卷及操作（16*14*14）->(32*12*12)
            nn.ReLU(),  # 激活函数
            nn.AvgPool2d(2, 2)  # 最大值池化操作(32*12*12) -> (32*6*6)
        )
        self.classifer = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    # 定义网络前向传播途径
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图层
        output = self.classifer(x)
        return output


# 输出网络结构
myconvnet = MyConvnet()


print(myconvnet)

# 定义网络的训练过程函数
def train_model(model, traindataloader, train_rate, criterion, optimizer, num_epoch=25):
    """

    :param model: 网络模型
    :param traindataloader:训练数据集，切分为训练集和验证集
    :param train_rate: 训练集百分百
    :param criterion: 损失函数
    :param optimizer: 优化方法
    :param num_epoch: 训练轮数
    :return:
    """
    # 计算训练使用的batch数量
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)  # 返回浮点数x的四舍五入值
    # 复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)
        # 每个epoch有两个训练阶段
        train_loss = 0.0
        train_correct = 0
        train_num = 0
        val_loss = 0.0
        val_correct = 0
        val_num = 0
        for step, (b_x, b_y) in enumerate(traindataloader):
            if step < train_batch_num:
                model.train()  # 设置模式为训练模式
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)  # 返回指定维度最大值的序号
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)  # b_x.size(0):取出第一个维度的数字，这里是64;.item()获得张量中的元素值
                train_correct += torch.sum(pre_lab == b_y.data)  # 将预测值与标签值相等的数累加
                train_num += b_x.size(0)
            else:
                model.eval()  # 设置模式为评估模式
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_correct += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        # 计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_correct.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correct.double().item() / val_num)
        print('{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))  # -1表示获取最后一个元素
        print('{} Val Loss:{:.4f} Val Acc:{:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
        # 拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    # 使用最好模型的参数
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={"epoch": range(num_epoch),
              "train_loss_all": train_loss_all,
              "val_loss_all": val_loss_all,
              "train_acc_all": train_acc_all,
              "val_acc_all": val_acc_all})
    for epoch in range(num_epoch):
        print('Epoch {}: Train Acc: {:.4f} Val Acc: {:.4f}'.format(
            epoch, train_acc_all[epoch], val_acc_all[epoch]))
    return model, train_process


# 对模型进行训练
#optimizer = torch.optim.Adam(myconvnet.parameters(), lr=0.0003)
optimizer = SGD(myconvnet.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()  # 损失函数
myconvnet, train_pross = train_model(myconvnet, train_loader, 0.8, criterion, optimizer, num_epoch=25)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_pross.epoch, train_pross.train_loss_all, "ro-", label="Train loss")
plt.plot(train_pross.epoch, train_pross.val_loss_all, "bs-", label="Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(train_pross.epoch, train_pross.train_acc_all, "ro-", label="Train acc")
plt.plot(train_pross.epoch, train_pross.val_acc_all, "bs-", label="Val acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.savefig("loss & acc")
plt.show()


# 对测试集进行预测，并可视化预测效果
myconvnet.eval()
output = myconvnet(test_data_x)
pre_lab = torch.argmax(output, 1)  # 返回预测值标签：torch.Size([64, 1, 28, 28])
acc = accuracy_score(test_data_y, pre_lab)  # 分类准确率分数是指所有分类正确的百分比
print("在测试集上的预测精度为：", acc)

# 计算混淆矩阵并可视化
conf_mat = confusion_matrix(test_data_y, pre_lab)
df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
heatmap = sns.heatmap(df_cm)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha="right")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('heatmap')
plt.show()



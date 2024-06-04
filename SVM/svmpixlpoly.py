import numpy as np
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt




# 使用FashionMNIST数据，准备训练数据集
train_data = FashionMNIST(
    root='../data/FashionMNIST',  # 数据路径
    train=True,  # 只使用训练数据集
    transform=transforms.ToTensor(),
    download=False
)

# 准备测试数据集
test_data = FashionMNIST(
    root='../data/FashionMNIST',  # 数据路径
    train=False,  # 使用测试数据集
    transform=transforms.ToTensor(),
    download=False
)


# 提取像素值
pixel_values = []
labels = []
for image, label in train_data:
    # image是一个[C, H, W]张量，其中C是通道数，H是高度，W是宽度
    pixels = image.numpy().flatten()  # 将图像张量转换为一维数组
    pixel_values.append(pixels)
    labels.append(label)

# 提取测试数据集的像素值和标签
test_pixel_values = []
test_labels = []
for image, label in test_data:
    pixels = image.numpy().flatten()  # 将图像张量转换为一维数组
    test_pixel_values.append(pixels)
    test_labels.append(label)



# 转换为numpy数组
pixel_values = np.array(pixel_values)
labels = np.array(labels)

test_pixel_values = np.array(test_pixel_values)
test_labels = np.array(test_labels)

# 创建SVM分类器实例，使用多项式核
clf = svm.SVC(kernel='poly',degree=3)

# 训练SVM分类器
clf.fit(pixel_values, labels)

# 使用训练好的分类器进行预测
predicted_labels = clf.predict(test_pixel_values)

# 计算准确率
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"SVM Classifier with Polynomial Kernel Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(test_labels, predicted_labels)

# 使用Seaborn绘制混淆矩阵的热图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# 打印分类报告
class_report = classification_report(test_labels, predicted_labels)
print(class_report)
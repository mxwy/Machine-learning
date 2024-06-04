import numpy as np
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# 定义一个函数来提取HOG特征
def extract_hog_features(dataset):
    hog_features = []
    for image, label in dataset:
        # 将图像转换为numpy数组并转换为灰度图像
        image = image.numpy().squeeze()
        #image = color.rgb2gray(image)
        features, hog_image = hog(image, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=True)
        hog_features.append(features)
        # 显示第一个图像的HOG可视化
    return np.array(hog_features)


# 使用FashionMNIST数据，准备训练数据集
train_data = FashionMNIST(
    root='../data/FashionMNIST',  # 数据路径
    train=True,  # 只使用训练数据集
    transform=transforms.ToTensor(),
    download=False
)

#加载测试集
test_data = FashionMNIST(
    root='../data/FashionMNIST',  # 数据路径
    train=False,  # 使用测试数据集
    transform=transforms.ToTensor(),
    download=False
)

#提取hog特征
train_hog_features = extract_hog_features(train_data)
test_hog_features = extract_hog_features(test_data)

# 获取训练和测试数据集的标签
train_labels = np.array(train_data.targets)
test_labels = np.array(test_data.targets)

#创建SVM分类器，使用多项式核
svm_classifier = svm.SVC(kernel='rbf')
# 训练SVM分类器
svm_classifier.fit(train_hog_features, train_labels)

# 在测试数据上进行预测
predicted_labels = svm_classifier.predict(test_hog_features)

# 计算准确率
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"SVM Classifier with RBF Kernel Accuracy: {accuracy:.2f}")

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
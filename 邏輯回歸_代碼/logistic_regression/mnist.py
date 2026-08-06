import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from logistic_regression import LogisticRegression

data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'mnist-demo.csv'))

# 繪圖設置
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10, 10))

#
for plot_index in range(numbers_to_display):
    # 讀取資料
    digit = data[plot_index:plot_index + 1].values
    digit_label = digit[0][0]
    digit_pixels = digit[0][1:]

    # 正方形的
    image_size = int(math.sqrt(digit_pixels.shape[0]))

    # 轉換成圖像形式
    frame = digit_pixels.reshape((image_size, image_size))
    
    # 展示圖像
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')
    plt.title(digit_label)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

# 訓練集劃分
pd_train_data = data.sample(frac=0.8)
pd_test_data = data.drop(pd_train_data.index)

# Ndarray数组格式
train_data = pd_train_data.values
test_data = pd_test_data.values

num_training_examples = 6000
x_train = train_data[:num_training_examples, 1:]
y_train = train_data[:num_training_examples, [0]]

x_test = test_data[:, 1:]
y_test = test_data[:, [0]]

# 訓練參數
max_iterations = 50000  
polynomial_degree = 0  
sinusoid_degree = 0  
normalize_data = True  

# 邏輯迴歸
logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree, normalize_data)

(thetas, costs) = logistic_regression.train(max_iterations)

pd.DataFrame(thetas)

# 要顯示多少個數字
numbers_to_display = 9

# 計算要顯示多少個數字
num_cells = math.ceil(math.sqrt(numbers_to_display))

# 讓圖片稍微大於預設大小
plt.figure(figsize=(10, 10))

# 遍歷thetas並顯示
for plot_index in range(numbers_to_display):
    # 提取數字資料
    digit_pixels = thetas[plot_index][1:]

    # 計算圖像大小（記住每個圖像都有正方形比例）
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    
    # 將圖像向量轉換為像素矩陣
    frame = digit_pixels.reshape((image_size, image_size))
    
    # 繪製數字矩陣
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')
    plt.title(plot_index)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# 繪製所有子圖
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

# 訓練情況
labels = logistic_regression.unique_labels
for index, label in enumerate(labels):
    plt.plot(range(len(costs[index])), costs[index], label=labels[index])

plt.xlabel('Gradient Steps')
plt.ylabel('Cost')
plt.legend()
plt.show()

# 測試結果
y_train_predictions = logistic_regression.predict(x_train)
y_test_predictions = logistic_regression.predict(x_test)

train_precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
test_precision = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100

print('Training Precision: {:5.4f}%'.format(train_precision))
print('Test Precision: {:5.4f}%'.format(test_precision))

# 要顯示多少個數字
numbers_to_display = 64

# 計算要顯示多少個數字
num_cells = math.ceil(math.sqrt(numbers_to_display))

# 讓圖片稍微大於預設大小
plt.figure(figsize=(15, 15))

# 遍歷測試集中的前幾個數字並繪製
for plot_index in range(numbers_to_display):
    # 提取數字資料
    digit_label = y_test[plot_index, 0]
    digit_pixels = x_test[plot_index, :]
    
    # 預測標籤
    predicted_label = y_test_predictions[plot_index][0]

    # 計算圖像大小（記住每個圖像都有正方形比例）
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    
    # 將圖像向量轉換為像素矩陣
    frame = digit_pixels.reshape((image_size, image_size))
    
    # 繽製數字矩陣
    color_map = 'Greens' if predicted_label == digit_label else 'Reds'
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap=color_map)
    plt.title(predicted_label)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# 繪製所有子圖
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
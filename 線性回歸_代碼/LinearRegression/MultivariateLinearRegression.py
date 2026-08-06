import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加詳細的導入檢查
try:
    import plotly
    import plotly.graph_objs as go
    import plotly.offline as pyo
    print(f"Plotly 版本: {plotly.__version__}")
    print("Plotly 導入成功")
except ImportError as e:
    print(f"Plotly 導入失敗: {e}")
    exit()

# 移除筆記本模式初始化，因為不在 Jupyter 中運行
# pyo.init_notebook_mode(connected=True)

from linear_regression import LinearRegression

print("開始加載數據...")
data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'world-happiness-report-2017.csv'))
print(f"數據形狀: {data.shape}")

train_data = data.sample(frac=0.8, random_state=42)  # 添加隨機種子確保結果可重現
test_data = data.drop(train_data.index)

input_param_name_1 = 'Economy..GDP.per.Capita.'
input_param_name_2 = 'Freedom'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name_1, input_param_name_2]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1, input_param_name_2]].values
y_test = test_data[[output_param_name]].values

print(f"訓練集大小: {x_train.shape}")
print(f"測試集大小: {x_test.shape}")

# Configure the plot with training dataset
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker=dict(
        size=8,
        opacity=0.8,
        color='blue',
        line=dict(
            color='rgb(255, 255, 255)',
            width=1
        ),
    )
)

plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker=dict(
        size=8,
        opacity=0.8,
        color='red',
        line=dict(
            color='rgb(255, 255, 255)',
            width=1
        ),
    )
)

plot_layout = go.Layout(
    title='Data Sets',
    scene=dict(
        xaxis=dict(title=input_param_name_1),
        yaxis=dict(title=input_param_name_2),
        zaxis=dict(title=output_param_name)
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    width=800,
    height=600
)

plot_data = [plot_training_trace, plot_test_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)

# 修正: 不自動打開，只保存文件
current_dir = os.getcwd()
html_file_path = os.path.join(current_dir, '3d_scatter_initial.html')

try:
    pyo.plot(plot_figure, filename=html_file_path, auto_open=False)
    print(f"初始圖表已保存至: {html_file_path}")
    print("請手動用瀏覽器打開該 HTML 文件查看圖表")
except Exception as e:
    print(f"保存圖表時出錯: {e}")

# 訓練模型
print("\n開始訓練模型...")
num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0
sinusoid_degree = 0

linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

print('開始損失:', cost_history[0])
print('結束損失:', cost_history[-1])

# 顯示成本歷史
plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.grid(True)
plt.show()

print("正在生成預測結果...")

# 生成預測平面
predictions_num = 15

x_min = x_train[:, 0].min()
x_max = x_train[:, 0].max()
y_min = x_train[:, 1].min()
y_max = x_train[:, 1].max()

x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)

x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

# 創建網格用於 Surface 圖
X_mesh, Y_mesh = np.meshgrid(x_axis, y_axis)
Z_mesh = z_predictions.reshape((predictions_num, predictions_num))

# 使用 Surface 圖顯示預測平面
plot_predictions_surface = go.Surface(
    x=X_mesh,
    y=Y_mesh,
    z=Z_mesh,
    name='Prediction Surface',
    opacity=0.7,
    colorscale='Viridis',
    showscale=False
)

# 創建包含預測結果的圖表
plot_data_with_predictions = [
    plot_training_trace, 
    plot_test_trace, 
    plot_predictions_surface
]

plot_figure_final = go.Figure(data=plot_data_with_predictions, layout=plot_layout)

# 保存最終圖表，不自動打開
final_html_path = os.path.join(current_dir, '3d_scatter_with_predictions.html')

try:
    pyo.plot(plot_figure_final, filename=final_html_path, auto_open=False)
    print(f"\n最終圖表已保存至: {final_html_path}")
    print("請手動用瀏覽器打開該 HTML 文件查看完整的 3D 可視化結果")
    
    # 提供額外的顯示選項
    print("\n其他查看選項:")
    print("1. 直接雙擊 HTML 文件")
    print("2. 右鍵點擊文件 -> 使用瀏覽器打開")
    print("3. 在瀏覽器中按 Ctrl+O 打開文件")
    
except Exception as e:
    print(f"保存最終圖表時出錯: {e}")
    
    # 備用方案：使用 show() 方法
    print("嘗試使用 show() 方法...")
    try:
        plot_figure_final.show()
    except Exception as e2:
        print(f"show() 方法也失敗: {e2}")

print("\n代碼執行完成")
print("HTML 文件已生成，請查看上述路徑中的文件")
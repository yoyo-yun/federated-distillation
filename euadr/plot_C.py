# coding:UTF-8
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

x = 0.1
input_value = []
for i in range(0, 20):
    input_value.append(x)
    x += 0.05
print(input_value)
# input_value = [0.1,0.15, 0.2,0.25, 0.3,0.35 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
squares = [77.23, 77.23, 78.57, 78.69, 79.35,
           79.68, 79.98, 80.15,81.12, 80.14, 80.77, 81.25, 81.70,81.69,81.45,
           81.50, 81.50,81.47,81.48,81.47]

input_value = np.array(input_value)
squares = np.array(squares)
x_smooth = np.linspace(input_value.min(), input_value.max(), 300)
y_smooth = make_interp_spline(input_value, squares)(x_smooth)
print(input_value)
print(y_smooth)
plt.plot(x_smooth, y_smooth, linewidth=2)

# 设置图形的标题，并给坐标轴加上标签
plt.title(u"Different C In GAD Dataset", fontsize=24)
plt.xlabel("C", fontsize=14)
plt.ylabel("F1", fontsize=14)

# 设置刻度表标记的大小
plt.tick_params(axis="both", labelsize=14)
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# # from scipy.interpolate import spline
# from scipy.interpolate import make_interp_spline
# x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# y = np.array([78.02, 78.24, 79.35, 79.68, 80.66, 81.65, 81.23, 82.71, 81.89, 81.22])
#
#
# x_smooth = np.linspace(x.min(), x.max(), 300)
# y_smooth = make_interp_spline(x, y)(x_smooth)
# # x_new = np.linspace(x.min(), x.max(), 300)  # 300 represents number of points to make between T.min and T.max
# # y_smooth = spline(x, y, x_new)
# # 散点图
# plt.scatter(x, y, c='black', alpha=0.5)  # alpha:透明度) c:颜色
# # 折线图
# plt.plot(x, y, linewidth=1)  # 线宽linewidth=1
# plt.plot(x, y, c='dimgray', linewidth=0.9, label='示例')  # label：某个线条代表什么的标签
# plt.legend(loc='best')  # loc选项可以选择label的位置
# # 平滑后的折线图
# plt.plot(x_smooth, y_smooth, c='red')
#
# # 解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei黑体  FangSong仿宋
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.title("绘图", fontsize=24)  # 标题及字号
# plt.xlabel("X", fontsize=24)  # X轴标题及字号
# plt.ylabel("Y", fontsize=24)  # Y轴标题及字号
# plt.tick_params(axis='both', labelsize=14)  # 刻度大小
# # plt.axis([0, 1100, 1, 1100000])#设置坐标轴的取值范围
# plt.show()

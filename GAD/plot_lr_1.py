# coding:UTF-8
import matplotlib.pyplot as plt

# x = 0.1
# input_value = []
# for i in range(0, 20):
#     input_value.append(x)
#     x += 0.05
# print(input_value)
input_value = ['2e-6', '3e-6', '4e-6', '2e-5', '3e-5', '4e-5', '2e-4', '3e-4', '4e-4', '2e-3', '3e-3', '4e-3', '2e-2',
               '3e-2', '4e-2']
squares = [78.02, 78.22, 78.34, 79.12, 79.35,
           79.68, 80.66, 80.98, 81.12, 81.65, 81.23, 82.45, 82.71, 82.57, 82.14, ]
# 81.89, 81.22,80.88,79.99,79.56]
plt.plot(input_value, squares, linewidth=2)

# 设置图形的标题，并给坐标轴加上标签
plt.title(u"GAD Dataset (Stacking)", fontsize=24)
plt.xlabel("Lamda", fontsize=14)
plt.ylabel("F1", fontsize=14)

# 设置刻度表标记的大小
plt.tick_params(axis="both", labelsize=14)
plt.show()

import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

waters = ['1', '2', '3', '4', '5', '6', '7']
# data = 33.36
# buy_number = []
# for i in range(1, 8):
#     buy_number.append(i * data)

buy_number = [20,30,35,41,50,55,60]
plt.xlabel("训练轮次", fontsize=14)
plt.ylabel("时间/min", fontsize=14)
plt.bar(waters, buy_number)
# plt.title('GAD关系抽取任务上传参数量监测')
plt.savefig('tiem_plot.png', transparent=True)
plt.show()

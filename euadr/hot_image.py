import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.random.rand(4,3)
fig, ax = plt.subplots(figsize = (9,9))
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(pd.DataFrame(np.round(a,2), columns = ['a', 'b', 'c'], index = range(1,5)), 
                annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
#sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
#            square=True, cmap="YlGnBu")
ax.set_title('二维数组热力图', fontsize = 18)
ax.set_ylabel('数字', fontsize = 18)
ax.set_xlabel('字母', fontsize = 18)
plt.show()
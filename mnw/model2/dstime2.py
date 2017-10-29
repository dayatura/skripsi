import glob
import numpy as np


files = glob.glob('*.json')
data = []

for file in files:
	detail  = file.split('_')
	detail[2] = detail[2].split('.')[0]
	data.append(detail)

print(data)
time = np.array(data)

np.savez('train_op_time2.npz', time=time)
### visualize

# model_1 = filter(lambda data: data[0] == 'model', data)
# model_2 = filter(lambda data: data[0] == 'model2', data)

# n_groups = 7

# means_men =  map(lambda x: x[2], model_1)
# std_men = (2, 3, 4, 1, 2)

# means_women = map(lambda x: x[2], model_2)
# std_women = (3, 5, 2, 3, 3)

# fig, ax = plt.subplots()

# index = np.arange(n_groups)
# bar_width = 0.35

# opacity = 0.4
# error_config = {'ecolor': '0.3'}

# rects1 = plt.bar(index, means_men, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  yerr=std_men,
#                  error_kw=error_config,
#                  label='Men')

# rects2 = plt.bar(index + bar_width, means_women, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  yerr=std_women,
#                  error_kw=error_config,
#                  label='Women')

# plt.xlabel('Group')
# plt.ylabel('Scores')
# plt.title('Scores by group and gender')
# plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E'))
# plt.legend()

# plt.tight_layout()
# plt.show()

import numpy
import matplotlib.pyplot as plt

################################## Plot model training process

data = numpy.load('result_optimizer.npz')
model_acc = data['model_acc']
model_loss = data['model_loss']
optimizer = data['optimizer']

plt.figure(1)
for op in range(len(optimizer)):
	plt.plot(model_acc[op])
plt.title( 'model accuracy' )
plt.ylabel( 'accuracy' )
plt.xlabel( 'epoch' )
plt.legend(optimizer, loc= 'best' )

plt.figure(2)
for op in range(len(optimizer)):
	plt.plot(model_loss[op])
plt.title( 'model loss' )
plt.ylabel( 'loss' )
plt.xlabel( 'epoch' )
plt.legend(optimizer, loc= 'best' )
# plt.show()

###################################### plot model accuracy

data = numpy.load('op_acc.npz')
model_acc = data['model_acc']

## visualize

optimizer = [str(i[0]) for i in model_acc]
acc_mean = [float(i[1]) for i in model_acc]
acc_std = [float(i[2]) for i in model_acc]
time = [float(i[3]) for i in model_acc]
n_groups = len(optimizer)

print(model_acc)

fig, ax = plt.subplots()

index = numpy.arange(n_groups)
bar_width = 0.45
opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, acc_mean, bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=acc_std,
                 error_kw=error_config,
                 label='Men')

plt.xlabel('Optimzer')
plt.ylabel('Accuracy')
plt.title('Model Accuracy by Optimzer')
plt.xticks(index + bar_width / 2, optimizer)
# plt.legend()
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)

plt.tight_layout()

######## plot model training time

fig, ax = plt.subplots()

index = numpy.arange(n_groups)
bar_width = 0.45
opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, time, bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='Men')

plt.xlabel('Optimzer')
plt.ylabel('Time (s)')
plt.title('Training Time by Optimzer')
plt.xticks(index + bar_width / 2, optimizer)
# plt.legend()
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)

plt.tight_layout()











plt.show()
import numpy
import matplotlib.pyplot as plt

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
plt.show()
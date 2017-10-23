import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#load data
dataset = numpy.load('aksara_sunda.npz')
X = dataset['data']
y = dataset['label']

#reshape to be [samples][width][height][chanels]

X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')

# X = X[0:10]
# y = y[0:10]

#normalize the data
X = X / 255

# one hot encode output
y = np_utils.to_categorical(y)
num_classes = y.shape[1]

def baseline_model(optimizer='adam'):
	# create model
	model = Sequential()
	model.add(Convolution2D(32, (6, 6), input_shape=(28, 28, 1), activation= 'relu' ))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# ############ k-fold validation
# estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=10, batch_size=1, verbose=1)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seedt)
# result = cross_val_score(estimator, X, y, cv=kfold)
# # numpy.savez('result_kfold.npz', result=result)
# # print(result)
# print("Accuracy: %.2f%% (%.2f%%)" % (result.mean()*100, result.std()*100))

############### split valdation
# build model
# model = baseline_model()
# # fit model
# history = model.fit(X, y, validation_split=0.33, nb_epoch=10, batch_size=1, verbose=2)


#### multiple optimizer
optimizer = ['adam','rmsprop','sgd','adagrad','adadelta','adamax','nadam']
model_acc = []
model_loss = []
for op in optimizer:
	model = baseline_model(op)
	history = model.fit(X, y, nb_epoch=10, batch_size=1, verbose=1)
	model_acc.append(history.history['acc'])
	model_loss.append(history.history['loss'])
	#serialize model to JSON
	model_json = model.to_json()
	model_name = "model_" + op + ".json"
	with open(model_name) as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	weights_name = "weights_" + op + ".h5"
	model.save_weights(weights_name)
	print("Saved model" + weights_name + " to disk")

numpy.savez('result_optimizer.npz', model_acc=model_acc, model_loss=model_loss, optimizer=optimizer)



# #serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
# 	json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

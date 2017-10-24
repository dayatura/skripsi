import numpy
import os
import sys
from PIL import Image
import PIL.ImageOps as ImOps
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

model_name = "mnw/model2_adam_121.20404028892517.json"
model_weight = 'mnw/weights2_adam_121.20404028892517.h5'




if (len(sys.argv) == 1):
	filename = '../gambar/resized/kertas-9_02.jpg'
else:
	script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
	rel_path = sys.argv[1]
	filename = os.path.join(script_dir, rel_path)
# load model and weight
json_file = open( model_name ,  'r' )
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_weight)
print("Loaded model from disk")

# preprocess data
img = Image.open(filename)
inverted_image = ImOps.invert(img)
data = np.array(inverted_image.convert('L'))
data = data.reshape(1, 28, 28, 1).astype('float32')
data = data / 255

# predict 
result = np.argmax(loaded_model.predict(data))
print(result)

# show image
data = data.reshape(1, 28, 28)
plt.subplot(111)
plt.imshow(data[0], cmap=plt.get_cmap('gray'))
plt.axis('off')
title = "Ini merupakan angka " + str(result)
plt.title( title)
plt.show()



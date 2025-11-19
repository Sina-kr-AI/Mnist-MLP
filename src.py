from keras import datasets, utils, models, layers, optimizers, losses
import numpy as np
import json
import matplotlib.pyplot as plt

# Data Preparation:

(train_img,train_labels),(test_img,test_labels)= datasets.mnist.load_data()

# Preprocessing:

train_x= train_img.reshape(60000,784)
test_x= test_img.reshape(10000,784)

# Normalizing (0-1):

train_x= train_x.astype('float32')/255
test_x= test_x.astype('float32')/255

Y_train = utils.to_categorical(train_labels)
Y_test = utils.to_categorical(test_labels)

# Import Our Model:

file_path="C:/Users/User/Desktop/machinlearning/vscode scripts/mnist-mlp practice/mnist-mlp scripts/model_config.json"

with open(file_path,'r') as f:
    model_config=json.load(f)

my_model=models.Sequential.from_config(model_config)

# Train Model:

history = my_model.fit(train_x, Y_train, batch_size=128, epochs=50, validation_split=0.2)

# Evaluation on Test Data:

test_loss = my_model.evaluate(test_x, Y_test)
print("test loss: ", test_loss)

test_labels_p = my_model.predict(test_x)
test_labels_p = np.argmax(test_labels_p, axis=1)

n = 0
f, axs = plt.subplots(1,10,figsize=(15,15))
for i in range(len(test_labels)):
    if n >= 10:
      break
    if (test_labels_p[i] != test_labels[i]):
      axs[n].imshow(test_img[i], cmap='gray')
      axs[n].set_title(f'{test_labels[i]} -> {test_labels_p[i]}')
      axs[n].axis('off')
      n = n+1
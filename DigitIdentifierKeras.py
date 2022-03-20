import PIL
import tensorflow
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(train_images,train_labels), (test_images, test_labels) = data.load_data()


model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(train_images,train_labels, epochs=7)
model.save("DigitCheck.h5")

model = keras.models.load_model("DigitCheck.h5")

test_loss, test_acc = model.evaluate(test_images,test_labels)
print(test_acc)

predictions = model.predict(test_images)




plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)

    plt.xlabel("Actual: "+str(test_labels[i]))
    plt.title("Prediction: "+str(np.argmax(predictions[i])))
    plt.show()


an_image = PIL.Image.open("7.png")
image_sequence = an_image.getdata()
image_array = np.array(image_sequence)

print(image_array)
image_array.reshape( 28,28)
print(type(test_images[0]))

print(type(image_array))
print(len(image_array))
predictions = model.predict([image_array])
print(predictions)
plt.grid(False)
plt.imshow(image_array, cmap=plt.cm.binary)
plt.title("Prediction: "+str(np.argmax(predictions[0])))
plt.show()
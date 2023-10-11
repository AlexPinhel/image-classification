import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import numpy as np



#img = image.load_img("basedata/training/happy/toto.png")
#cv2.imread("basedata/training/happy/1.jpeg").shape()

train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory('basedata/training/',
                                          target_size = (200,200),
                                          batch_size = 3,
                                          class_mode = 'binary')

validation_dataset = validation.flow_from_directory('basedata/validation/',
                                          target_size = (200,200),
                                          batch_size = 3,
                                          class_mode = 'binary')

# Configuration du modèle avec les couches
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape = (200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    ##
    tf.keras.layers.Flatten(),
    ##
    tf.keras.layers.Dense(512, activation='relu'),
    ##
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss= 'binary_crossentropy',
              optimizer= tf.keras.optimizers.legacy.RMSprop(
                  learning_rate=0.001
              ),
              metrics=['accuracy'])

# Entrainement du modèle
model_fit = model.fit(train_dataset,
                      steps_per_epoch= 3,
                      epochs=30,
                      validation_data=validation_dataset)


# Utilisation du modèle pour prédire sur nos données

dir_path = os.getcwd() + '/basedata/testing'

print("Directory path is: " + dir_path)

for i in os.listdir(dir_path):
    if not i.startswith('.'):
        file = dir_path+'//'+ i
        img = image.load_img(file, target_size=(200,200))
        plt.imshow(img)
        plt.show()

        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])
        val = model.predict(images)
        if val == 0:
            print(file + " ==> You are not happy")
        else:
            print(file + " ==> You are happy")
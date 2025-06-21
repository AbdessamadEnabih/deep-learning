import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = 'clothes_classifier.keras'

# Ensure the models directory exists
if not os.path.exists('./models'):
    os.makedirs('./models')
    
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

if os.path.exists(f'./models/{MODEL_NAME}'):
    print("Loading existing model...")
    model = tf.keras.models.load_model(f'./models/{MODEL_NAME}')
else:
    print("No existing model found, creating a new one...")

    index = 0

    np.set_printoptions(linewidth=320)

    # Print the label and image
    print(f'LABEL: {training_labels[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n\n{training_images[index]}\n\n')



    # Normalize the pixel values of the train and test images
    training_images  = training_images / 255.0
    test_images = test_images / 255.0

    # Build the classification model
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(28,28)),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(128, activation='relu'), 
        tf.keras.layers.Dense(10, activation='softmax')
    ])


    class custom_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') > 0.98:
                print("\nReached 98% accuracy so cancelling training!")
                self.model.stop_training = True
        def on_train_begin(self, logs=None):
            print("\nTraining started...")

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=50, callbacks=[custom_callback()])

    model.evaluate(test_images, test_labels)
    
    model.save(f'./models/{MODEL_NAME}')
    print("Model saved as: ", MODEL_NAME )

# Define class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for test_index in range(len(test_images)):
    if test_index >= 10:
        break
    predictions = model.predict(test_images[test_index:test_index+1])
    predicted_class = np.argmax(predictions[0])
    actual_class = test_labels[test_index]

    print(f'Predicted: {class_names[predicted_class]}')
    print(f'Actual: {class_names[actual_class]}')


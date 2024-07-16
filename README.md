# Image Classification Using CIFAR-10

## Project Overview

This project involves building an image classification model using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The model is implemented using TensorFlow and Keras, leveraging a Convolutional Neural Network (CNN) architecture to classify images into the predefined categories.

## Dataset

The CIFAR-10 dataset includes the following classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Each class has 6,000 images. The dataset is divided into 50,000 training images and 10,000 test images.

## Prerequisites

Make sure you have the following installed:
- Python 3.6 or higher
- TensorFlow
- NumPy
- Matplotlib

You can install the necessary packages using pip:
```bash
pip install tensorflow numpy matplotlib
```

## Project Structure

```
.
├── README.md
├── cifar10_classification.py
└── requirements.txt
```

### `cifar10_classification.py`

This script contains the entire code to build, train, and evaluate the CNN model.

### `requirements.txt`

This file lists the required Python packages.

## Code Explanation

### Import Libraries
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
```

### Load and Preprocess Data
```python
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
```

### Define the CNN Model
```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
```

### Compile the Model
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

### Train the Model
```python
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```

### Evaluate the Model
```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

### Visualize Training History
```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
```

### Make Predictions and Plot Results
```python
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel(f"{class_names[predicted_label]} ({class_names[true_label[0]]})", color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i][0]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
```

## Running the Project

1. Clone the repository or download the `cifar10_classification.py` script.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python cifar10_classification.py
   ```

## Results

The model is trained for 10 epochs. The accuracy and loss during training and validation are plotted. The script also includes a function to visualize the model's predictions on the test dataset, highlighting the predicted and true labels of images.

## Future Work

- **Data Augmentation**: Implement data augmentation to improve model generalization.
- **Advanced Architectures**: Explore deeper architectures like ResNet or Inception for better performance.
- **Transfer Learning**: Utilize pre-trained models on larger datasets (e.g., ImageNet) for transfer learning.

## References

- [TensorFlow Documentation](https://www.tensorflow.org/tutorials/images/cnn)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

This README provides a comprehensive overview of the project, from setting up the environment to running the model and evaluating its performance. Feel free to modify and expand it based on your project's specific details and requirements.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing import image

# Define the image size
img_width, img_height = 200, 200

# Preprocess and data augmentation of the images
train_img_pp = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
    )
test_img_pp = ImageDataGenerator(rescale=1./255)

# Importing directories containing images of male and female and storing it in a variable
train_dir = r'Dataset\train'
test_dir = train_dir = r'Dataset\validation'

# Create generators for training and validation sets
train_generator = train_img_pp.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=10,
    class_mode='binary'
)
test_generator = test_img_pp.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=10,
    class_mode='binary'
)

# Model construction
model = Sequential()

# CNN
# First layer
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))

# Second layer
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

#Third layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten layer
model.add(Flatten())

# Dense layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output Layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# This callback stops training if the validation loss doesn’t improve for a specified number of epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fitting the model
model.fit(train_generator, epochs=5, validation_data=test_generator, callbacks=[early_stopping])

# Evaluating the test and train Accuracy
test_loss, test_accuracy = model.evaluate(test_generator)
train_loss, train_accuracy = model.evaluate(train_generator)
print(f'Test Accuracy: {test_accuracy}')
print(f'Train Accuracy: {train_accuracy}')

model.save('m_f_prediction.h5')
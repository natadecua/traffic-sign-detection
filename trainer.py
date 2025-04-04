import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

# Data Preprocessing
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Define the training and validation generators using the ImageDataGenerator
train_gen = data_gen.flow_from_directory(
    r'C:\Users\chloe\Downloads\LBYCPF_Project\dataset',  # Replace with correct path
    target_size=(64, 64),  # Resizing to 64x64
    batch_size=32, 
    class_mode='categorical', 
    subset='training'
)

val_gen = data_gen.flow_from_directory(
    r'C:\Users\chloe\Downloads\LBYCPF_Project\dataset',  # Replace with correct path
    target_size=(64, 64),  # Resizing to 64x64
    batch_size=32, 
    class_mode='categorical', 
    subset='validation'
)

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(3, activation='softmax')  # 3 classes 
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save Model
model.save('road_sign_model.h5')
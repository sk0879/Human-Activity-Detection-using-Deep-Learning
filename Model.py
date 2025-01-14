import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

# Define the model
def build_model(input_shape, num_classes):
    """
    Build a CNN-LSTM model for activity detection using VGG16 as the base.
    
    Parameters:
    - input_shape: Shape of the input video (frames).
    - num_classes: Number of output classes (activity categories).
    
    Returns:
    - model: Compiled CNN-LSTM model.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the layers of VGG16

    model = Sequential([
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(64, return_sequences=False),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(data_dir, model_path):
    """
    Train the model on the video data.
    
    Parameters:
    - data_dir: Directory containing the training data (frames).
    - model_path: Path to save the trained model.
    """
    datagen = ImageDataGenerator(rescale=1.0/255)  # Normalize the images
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )
    
    model = build_model(input_shape=(None, 64, 64, 3), num_classes=len(train_generator.class_indices))
    model.fit(train_generator, epochs=10)
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Evaluate the model
def evaluate_model(model_path, data_dir):
    """
    Evaluate the model on test data.
    
    Parameters:
    - model_path: Path to the trained model.
    - data_dir: Directory containing the test data.
    """
    model = tf.keras.models.load_model(model_path)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    test_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )
    
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict activity from video
def predict_activity(model_path, video_path):
    """
    Predict activities in a video using the trained model.
    
    Parameters:
    - model_path: Path to the trained model.
    - video_path: Path to the video to make predictions on.
    """
    model = tf.keras.models.load_model(model_path)
    
    cap = cv2.VideoCapture(video_path)
    frame_sequence = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_resized = cv2.resize(frame, (64, 64))
        frame_sequence.append(frame_resized / 255.0)
        
        if len(frame_sequence) == 30:  # Predict after 30 frames
            input_data = np.expand_dims(frame_sequence, axis=0)  # Add batch dimension
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            print(f"Predicted Activity: Class {predicted_class}")
            frame_sequence = []
    
    cap.release()

# Preprocess video (Extract frames from video)
def preprocess_video(video_path, output_dir, frame_rate=30):
    """Extract frames from a video at the specified frame rate."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    count = 0
    success, frame = cap.read()
    while success:
        if count % frame_rate == 0:
            output_path = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(output_path, frame)
        success, frame = cap.read()
        count += 1
    cap.release()
    print(f"Frames saved in {output_dir}")

# Example usage of all functions
if __name__ == "__main__":
    # Preprocess video into frames
    preprocess_video("data/raw/sample_video.mp4", "data/processed_frames")
    
    # Train the model
    train_model(data_dir='data/train', model_path='models/activity_model.h5')
    
    # Evaluate the model
    evaluate_model(model_path='models/activity_model.h5', data_dir='data/test')
    
    # Make predictions on a video
    predict_activity(model_path='models/activity_model.h5', video_path='data/sample_video.mp4')

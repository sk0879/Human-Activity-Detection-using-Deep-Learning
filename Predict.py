import tensorflow as tf
import cv2
import numpy as np
import argparse

# Function to load the pretrained model
def load_model(model_path):
    """Load the pretrained model from the specified path."""
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess frames and make predictions
def predict_activity(model, video_path):
    """Predict human activity in the provided video."""
    cap = cv2.VideoCapture(video_path)  # Open the video file
    frame_sequence = []  # List to store frames

    while cap.isOpened():
        success, frame = cap.read()  # Read a frame from the video
        if not success:
            break  # Break the loop if no more frames are available
        
        # Preprocess the frame: resize to the model input size (e.g., 64x64)
        frame_resized = cv2.resize(frame, (64, 64))  # Resize frame to 64x64
        frame_sequence.append(frame_resized / 255.0)  # Normalize and append to sequence
        
        # If we have 30 frames in the sequence, we make a prediction
        if len(frame_sequence) == 30:  # For example, 30 frames per sequence
            input_data = np.expand_dims(frame_sequence, axis=0)  # Add batch dimension
            prediction = model.predict(input_data)  # Predict activity
            predicted_class = np.argmax(prediction)  # Get the predicted class index
            print(f"Predicted Activity: Class {predicted_class}")
            frame_sequence = []  # Reset the frame sequence after prediction

    cap.release()  # Release the video capture object

# Command-line interface for running the script
if __name__ == "__main__":
    # Set up argument parser to allow command-line input
    parser = argparse.ArgumentParser(description="Predict human activity from a video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--model_path", type=str, default="models/activity_model.h5", help="Path to the pretrained model.")
    args = parser.parse_args()  # Parse the command-line arguments

    # Load the model and make predictions
    model = load_model(args.model_path)  # Load the pretrained model
    predict_activity(model, args.video_path)  # Predict activity in the video

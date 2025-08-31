import numpy as np
import cv2
import os

class CNNLSTMModel:
    def __init__(self, model_path=None, sequence_length=16, image_size=(128, 128)):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.frame_sequence = []
        print("CNN+LSTM model initialized")
    
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, self.image_size)
        return frame
    
    def add_frame(self, frame):
        processed_frame = self.preprocess_frame(frame)
        
        if len(self.frame_sequence) >= self.sequence_length:
            self.frame_sequence.pop(0)
        
        self.frame_sequence.append(processed_frame)
    
    def predict(self):
        if len(self.frame_sequence) < self.sequence_length:
            return "no_action", 0.0
        
        return "no_action", 0.0
    
    def reset_sequence(self):
        """Reset the frame sequence"""
        self.frame_sequence = []
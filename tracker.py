import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import pickle
import os

class EliteShotEngine:
    def __init__(self, model_path='yolov8x.pt'):
        # 1. Use the Extra Large Model (As requested)
        print(f"Loading Professional Model: {model_path}...")
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """
        Runs the Heavy AI Inference on a list of frames.
        Returns a list of Detections.
        """
        batch_size = 20 
        detections_list = []
        
        print("🚀 Starting Inference on frames...")
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            results = self.model(batch, verbose=False)
            
            for result in results:
                # Convert to Supervision format
                det = sv.Detections.from_ultralytics(result)
                # Filter strictly for Person (0)
                det = det[det.class_id == 0]
                # Update Tracker
                det = self.tracker.update_with_detections(det)
                detections_list.append(det)
                
            print(f"Processed batch {i}/{len(frames)}")
            
        return detections_list

    def get_player_color(self, frame, bbox):
        """
        Refined Logic:
        1. Crop Player.
        2. Crop Torso (Top 50%) to avoid shorts/socks.
        3. Mask Green Pitch.
        4. Calculate Average Color.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Safe crop
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        player_img = frame[y1:y2, x1:x2]
        
        if player_img.size == 0: return None

        # --- TORSO CROP (Crucial for Team ID) ---
        # We only look at the top 50% of the bounding box (The Jersey)
        player_height = y2 - y1
        torso_img = player_img[0:int(player_height * 0.5), :]

        # --- GREEN MASKING ---
        hsv = cv2.cvtColor(torso_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 0, 0])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)
        
        # Keep only non-green pixels
        torso_content = cv2.bitwise_and(torso_img, torso_img, mask=mask_inv)
        
        pixels = torso_content.reshape(-1, 3)
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

        if len(pixels) < 50: return None # Not enough jersey visible

        return pixels.mean(axis=0)
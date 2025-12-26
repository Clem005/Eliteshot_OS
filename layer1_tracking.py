import cv2
import pickle
import numpy as np
from ultralytics import YOLO
import supervision as sv

def run_layer_1(video_path, output_stub_path):
    # --- CONFIGURATION ---
    # We use YOLOv8x (Extra Large) as requested for maximum accuracy
    MODEL_NAME = 'yolov8x.pt'
    BATCH_SIZE = 20 # As per your plan
    
    print(f"🚀 LAYER 1 START: Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    # Initialize ByteTracker
    # track_thresh: Confidence threshold to keep a track
    # match_thresh: Threshold for matching detection to track
    # UPDATED for newer Supervision versions
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25, 
        lost_track_buffer=30, 
        minimum_matching_threshold=0.8, 
        frame_rate=30
    )    
    # Open Video
    cap = cv2.VideoCapture(video_path)
    frames = []
    print("📥 Reading video into memory (for batch processing)...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    total_frames = len(frames)
    print(f"✅ Loaded {total_frames} frames.")

    # Storage for results
    # Structure: { frame_index: [ (bbox, class_id, track_id), ... ] }
    final_tracks = {}

    # --- INFERENCE LOOP (BATCHED) ---
    print("🧠 Starting Heavy Inference (This will take time on i3)...")
    
    for i in range(0, total_frames, BATCH_SIZE):
        # Prepare Batch
        batch = frames[i : i + BATCH_SIZE]
        
        # 1. PREDICT (YOLOv8x)
        # conf=0.1 ensures we catch the ball even if confidence is low
        results = model.predict(batch, conf=0.1, verbose=False)
        
        # 2. TRACK (ByteTrack) - Process one by one inside the batch
        for j, result in enumerate(results):
            frame_index = i + j
            
            # Convert to Supervision Detections
            detections = sv.Detections.from_ultralytics(result)
            
            # Filter: Keep Persons (0) and Sports Ball (32)
            # Note: Checking for class 32 might vary depending on dataset versions, 
            # usually COCO class 32 is 'sports ball'.
            detections = detections[(detections.class_id == 0) | (detections.class_id == 32)]
            
            # Update Tracker
            # Note: ByteTrack usually only tracks people robustly. 
            # Ball tracking often needs a separate logic, but we will pass it through here.
            detections = tracker.update_with_detections(detections)
            
            # Save Data
            frame_data = []
            for k in range(len(detections)):
                bbox = detections.xyxy[k]
                class_id = detections.class_id[k]
                track_id = detections.tracker_id[k]
                
                # Store as a simple tuple to save space/memory
                frame_data.append({
                    "bbox": bbox.tolist(),
                    "class_id": int(class_id),
                    "track_id": int(track_id) if track_id is not None else -1
                })
            
            final_tracks[frame_index] = frame_data
        
        print(f"✅ Processed batch {i}/{total_frames}")

    # --- SAVE STUB ---
    print(f"💾 Saving results to {output_stub_path}...")
    with open(output_stub_path, 'wb') as f:
        pickle.dump(final_tracks, f)
    
    print("🎉 LAYER 1 COMPLETE. You never have to run this again for this video.")

if __name__ == "__main__":
    # Ensure input.mp4 exists!
    run_layer_1("input.mp4", "stubs/track_stub.pkl")
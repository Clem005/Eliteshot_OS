import cv2
import numpy as np
import pickle
import json
import pandas as pd
from sklearn.cluster import KMeans
from collections import deque

# --- CONFIGURATION ---
PITCH_WIDTH_METERS = 105
PITCH_HEIGHT_METERS = 68
FPS = 30

# ⚠️ CRITICAL: UPDATE THESE POINTS FOR YOUR VIDEO!
# Define the 4 corners of the PLAYABLE GRASS AREA in your specific video.
# Format: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
SRC_POINTS = np.float32([
    [100, 200],    # Point 1: Top-Left Corner of the pitch
    [1820, 200],   # Point 2: Top-Right Corner
    [1820, 1000],  # Point 3: Bottom-Right Corner
    [100, 1000]    # Point 4: Bottom-Left Corner
])

DST_POINTS = np.float32([
    [0, 0], 
    [PITCH_WIDTH_METERS, 0],
    [PITCH_WIDTH_METERS, PITCH_HEIGHT_METERS], 
    [0, PITCH_HEIGHT_METERS]
])

def is_inside_pitch(point, polygon_pts):
    """
    Returns True if the point (x,y) is strictly inside the pitch boundary.
    """
    poly = polygon_pts.astype(int)
    # measureDist=False returns +1 if inside, -1 if outside
    result = cv2.pointPolygonTest(poly, point, False)
    return result >= 0

def get_jersey_color(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w, _ = frame.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    img = frame[y1:y2, x1:x2]
    if img.size == 0: return None
    
    # Strict Torso Crop (15% to 50% of height) to avoid head/shorts
    height = y2 - y1
    width = x2 - x1
    img = img[int(height*0.15):int(height*0.5), int(width*0.2):int(width*0.8)]
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Mask Green Pitch
    mask_green = cv2.inRange(hsv, np.array([35,0,0]), np.array([85,255,255]))
    # Mask Dark colors (Refs/Coaches)
    mask_dark = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,50]))
    combined_mask = cv2.bitwise_or(mask_green, mask_dark)
    
    img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(combined_mask))
    pixels = img.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0,0,0], axis=1)]
    
    if len(pixels) < 30: return None 
    return pixels.mean(axis=0)

def draw_broadcast_label(img, text_lines, pos, bg_color=(0,0,0), text_color=(255,255,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    padding = 4
    line_height = 18
    x, y = pos
    
    # Calculate box dimensions
    max_w = 0
    total_h = len(text_lines) * line_height + padding
    for line in text_lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        if w > max_w: max_w = w
    w = max_w + (padding * 2)
    
    # Draw Glass Box
    overlay = img.copy()
    top_left = (x - w//2, y - total_h)
    bottom_right = (x + w//2, y)
    cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Draw Text
    current_y = top_left[1] + line_height
    for line in text_lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.putText(img, line, (x - tw // 2, current_y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)
        current_y += line_height

def analyze_match_broadcast(video_path, stub_path, output_json, output_video):
    print(f"📂 Loading Stubs from {stub_path}...")
    with open(stub_path, 'rb') as f: raw_tracks = pickle.load(f)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
    
    homography_matrix = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

    # --- PHASE 1: CAMERA MOTION ---
    print("🎥 Pass 1: Camera Motion...")
    camera_movement = [[0,0]] * total_frames
    old_gray = None
    curr_frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if old_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            camera_movement[curr_frame_idx] = np.mean(flow, axis=(0,1))
        old_gray = gray
        curr_frame_idx += 1
    cap.release()

    # --- PHASE 2: STRICT TEAM TRAINING ---
    print("🎨 Pass 2: Strict Team Color Training...")
    cap = cv2.VideoCapture(video_path)
    player_colors = []
    
    # Sample every 10th frame for speed/variety
    for i in range(0, min(300, total_frames), 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        if i in raw_tracks:
            for entity in raw_tracks[i]:
                if entity['class_id'] == 0: # Person
                    bbox = entity['bbox']
                    foot_x = (bbox[0] + bbox[2]) / 2
                    foot_y = bbox[3]
                    
                    # FILTER: Strict Pitch Boundary
                    if is_inside_pitch((foot_x, foot_y), SRC_POINTS):
                        c = get_jersey_color(frame, bbox)
                        if c is not None: player_colors.append(c)

    kmeans = None
    if len(player_colors) > 10:
        print(f"   Training on {len(player_colors)} clean samples...")
        kmeans = KMeans(n_clusters=2, n_init=10).fit(player_colors)
    else:
        print("   ⚠️ WARNING: Not enough valid samples found. Check SRC_POINTS.")

    # --- PHASE 2.5: BALL INTERPOLATION ---
    print("⚽ Pass 2.5: Ball Interpolation...")
    ball_positions = []
    for frame_num in range(total_frames):
        detected = False
        if frame_num in raw_tracks:
            for entity in raw_tracks[frame_num]:
                if entity['class_id'] == 32: # Ball
                    bbox = entity['bbox']
                    cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                    # Filter: Ball must be roughly inside pitch Y-bounds
                    if is_inside_pitch((cx, cy), SRC_POINTS):
                        ball_positions.append({'frame': frame_num, 'x': cx, 'y': cy})
                        detected = True
                        break
        if not detected:
            ball_positions.append({'frame': frame_num, 'x': np.nan, 'y': np.nan})

    df_ball = pd.DataFrame(ball_positions)
    df_ball = df_ball.interpolate().fillna(method='bfill')
    ball_lookup = {int(row['frame']): (row['x'], row['y']) for index, row in df_ball.iterrows()}

    # --- PHASE 3: BROADCAST RENDER ---
    print("🎬 Pass 3: Final Broadcasting...")
    cap = cv2.VideoCapture(video_path)
    final_output = []
    player_stats = {} 
    team_possession = {1: 0, 2: 0} 
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cam_x, cam_y = camera_movement[frame_idx] if hasattr(camera_movement[frame_idx], '__len__') else (0,0)
        frame_data = {"frame": frame_idx, "camera_move": [float(cam_x), float(cam_y)], "players": [], "ball": None}
        
        # Draw Camera Stats
        cv2.putText(frame, f"Cam X: {cam_x:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(frame, f"Cam X: {cam_x:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

        # --- PROCESS PLAYERS FIRST ---
        closest_dist = 9999
        closest_team = None
        
        # Get Ball Position for this frame
        current_ball_pos = None
        if frame_idx in ball_lookup:
            bx, by = ball_lookup[frame_idx]
            if not np.isnan(bx) and not np.isnan(by):
                current_ball_pos = (bx, by)

        if frame_idx in raw_tracks:
            for entity in raw_tracks[frame_idx]:
                if entity['class_id'] == 0:
                    bbox = entity['bbox']
                    tid = entity['track_id']
                    x1, y1, x2, y2 = map(int, bbox)
                    foot_x, foot_y = (x1+x2)/2, y2
                    
                    # 1. DELETE IF OUTSIDE PITCH
                    if not is_inside_pitch((foot_x, foot_y), SRC_POINTS):
                        continue

                    # 2. MATH
                    pt = cv2.perspectiveTransform(np.array([[[foot_x, foot_y]]], dtype='float32'), homography_matrix)[0][0]
                    meter_x, meter_y = pt[0], pt[1]

                    team_id = 0
                    if kmeans:
                        c = get_jersey_color(frame, bbox)
                        if c is not None: team_id = int(kmeans.predict([c])[0]) + 1
                    
                    if tid not in player_stats:
                        player_stats[tid] = {'last_pos': (meter_x, meter_y), 'dist': 0.0, 'speed_buffer': deque(maxlen=5)}
                    stats = player_stats[tid]
                    dist_frame = np.sqrt((meter_x - stats['last_pos'][0])**2 + (meter_y - stats['last_pos'][1])**2)
                    if dist_frame < 5.0: stats['dist'] += dist_frame
                    speed_kmh = (dist_frame * FPS) * 3.6
                    stats['speed_buffer'].append(speed_kmh)
                    avg_speed = sum(stats['speed_buffer']) / len(stats['speed_buffer'])
                    stats['last_pos'] = (meter_x, meter_y)

                    frame_data['players'].append({
                        "id": tid, "pos": [float(meter_x), float(meter_y)], "team": team_id,
                        "speed": float(avg_speed), "dist": float(stats['dist'])
                    })

                    # Possession Check
                    if current_ball_pos:
                        d_ball = np.sqrt((foot_x - current_ball_pos[0])**2 + (foot_y - current_ball_pos[1])**2)
                        if d_ball < closest_dist:
                            closest_dist = d_ball
                            closest_team = team_id

                    # 3. VISUALS (Player Layer)
                    color_vis = (200, 200, 200)
                    if team_id == 1: color_vis = (50, 50, 200) # Red-ish
                    if team_id == 2: color_vis = (200, 200, 50) # Blue-ish
                    
                    # Ellipse at feet
                    cv2.ellipse(frame, (int(foot_x), int(foot_y)), (16, 8), 0, 0, 360, (0,0,0), 3)
                    cv2.ellipse(frame, (int(foot_x), int(foot_y)), (16, 8), 0, 0, 360, color_vis, 2)
                    
                    # Floating Label
                    label_lines = [f"ID: {tid}"]
                    if avg_speed > 1.0:
                        label_lines.append(f"{avg_speed:.1f} km/h")
                        label_lines.append(f"{stats['dist']:.1f} m")
                    
                    draw_broadcast_label(frame, label_lines, (int(foot_x), y1 - 10), bg_color=(20, 20, 20))

        # --- DRAW BALL (TOP LAYER) ---
        if current_ball_pos:
            bx, by = current_ball_pos
            # Yellow Inverted Triangle
            tri_pts = np.array([[int(bx), int(by)-15], [int(bx)-8, int(by)-28], [int(bx)+8, int(by)-28]])
            cv2.drawContours(frame, [tri_pts], 0, (0, 0, 0), 2) # Outline
            cv2.drawContours(frame, [tri_pts], 0, (0, 255, 255), -1) # Fill
            
            # Save to JSON
            pt = cv2.perspectiveTransform(np.array([[[bx, by]]], dtype='float32'), homography_matrix)[0][0]
            frame_data['ball'] = {"pos": [float(pt[0]), float(pt[1])]}

        # --- DRAW STATS OVERLAY ---
        if closest_team and closest_dist < 60:
            team_possession[closest_team] += 1
        
        total_p = team_possession[1] + team_possession[2]
        if total_p > 0:
            p1 = (team_possession[1] / total_p) * 100
            p2 = (team_possession[2] / total_p) * 100
        else: p1, p2 = 50, 50

        overlay = frame.copy()
        cv2.rectangle(overlay, (width-300, height-80), (width, height), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"Team 1 Control: {p1:.0f}%", (width-280, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Team 2 Control: {p2:.0f}%", (width-280, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,200), 1, cv2.LINE_AA)

        out.write(frame)
        final_output.append(frame_data)
        
        frame_idx += 1
        if frame_idx % 30 == 0: print(f"   Broadcasting frame {frame_idx}/{total_frames}...")

    cap.release()
    out.release()
    with open(output_json, 'w') as f: json.dump(final_output, f)
    print(f"✅ DONE. Video: {output_video}")

if __name__ == "__main__":
    analyze_match_broadcast("input.mp4", "stubs/track_stub.pkl", "final_match_data.json", "final_output.mp4")
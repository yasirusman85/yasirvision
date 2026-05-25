import cv2
import numpy as np
import time
from datetime import datetime
import yasirvision as yv
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    success, img = cap.read()
    if not success:
        print("Error: Could not open camera.")
        return
    h, w, c = img.shape
    
    # Hand Detector
    detector = yv.HandDetector(maxHands=2, detectionCon=0.8)
    
    # Mode state:
    # 1: Hand Tracking
    # 2: Drawing Canvas
    # 3: Air Writing
    # 4: Gesture Control
    # 5: Finger Counter
    # 6: Invisibility Cloak
    # 7: Clock & Stopwatch
    # 8: Selfie Segmentation (Background Blur)
    # 9: Cyberpunk Face Mesh
    # 0: Virtual Keyboard
    mode = 1
    
    # State variables for Drawing (Mode 2)
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 255, 255),# White
        (0, 0, 0)       # Eraser
    ]
    colorIndex = 0
    drawColor = colors[colorIndex]
    brushThickness = 15
    eraserThickness = 50
    xp, yp = 0, 0
    imgCanvas = np.zeros((h, w, 3), np.uint8)
    prev_points = {"Left": (0, 0), "Right": (0, 0)}
    alpha = 0.35 # EMA smoothing factor
    last_selection_time = 0
    cooldown_delay = 0.5
    
    # State variables for Air Writing (Mode 3)
    imgAirCanvas = np.zeros((h, w, 3), np.uint8)
    xp_air, yp_air = 0, 0
    airColor = (255, 0, 255) # Pink
    fist_start_time = None
    clear_delay = 1.5
    
    # State variables for Invisibility Cloak (Mode 6)
    cloak_background = None
    cloak_capture_start = 0
    
    # State variables for Stopwatch (Mode 7)
    stopwatch_running = False
    stopwatch_start_time = 0
    stopwatch_elapsed_time = 0
    
    # Selfie Segmentation Setup (Mode 8)
    mp_selfie = mp.solutions.selfie_segmentation
    selfie = mp_selfie.SelfieSegmentation(model_selection=1)
    
    # Face Mesh Setup (Mode 9)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))
    conn_spec = mp_draw.DrawingSpec(thickness=1, color=(0, 255, 0))
    
    # Virtual Keyboard Setup (Mode 0)
    keys_layout = [
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
        ["SPACE", "BACK", "CLEAR"]
    ]
    
    class Key:
        def __init__(self, char, pos, size=(80, 80)):
            self.char = char
            self.pos = pos
            self.size = size
            
    keyboard_keys = []
    y_start = 140
    for row in keys_layout:
        x_start = 100
        for char in row:
            if char in ["SPACE", "BACK", "CLEAR"]:
                width = 250 if char == "SPACE" else 150
                keyboard_keys.append(Key(char, (x_start, y_start), (width, 65)))
                x_start += width + 20
            else:
                keyboard_keys.append(Key(char, (x_start, y_start), (80, 65)))
                x_start += 100
        y_start += 80
        
    typed_text = ""
    pinch_cooldown = 0
    
    print("=========================================================")
    print("      YASIRVISION 10-FEATURE UNIFIED TESTING SUITE       ")
    print("=========================================================")
    print("Instant Mode Switching Keys (1-9, 0):")
    print("  '1' : Hand Tracking (precision UP/DOWN status for both hands)")
    print("  '2' : Virtual Drawing Canvas (EMA smoothing, selection zone cooldown)")
    print("  '3' : Air Writing (Pink ink, 5 fingers=Eraser, Fist 1.5s=Auto-clear)")
    print("  '4' : Gesture Control (Dual hand volume pinch sliders & fists)")
    print("  '5' : Finger Counter (Individual counts + high-tech grand total card)")
    print("  '6' : Invisibility Cloak (Red substitution. Press 'b' to recapture)")
    print("  '7' : Clock & Gestures Stopwatch (Index up=Start, 2 fingers=Pause, Fist=Reset)")
    print("  '8' : Selfie Segmentation (Translucent local Gaussian background blur)")
    print("  '9' : Cyberpunk Face Mesh (Full 468 3D cybernetic tracking net)")
    print("  '0' : Virtual Interactive Keyboard (Hover to highlight, pinch to TYPE)")
    print("---------------------------------------------------------")
    print("Controls:")
    print("  'c' : Clear drawing canvas")
    print("  'b' : Recapture background (for Invisibility Cloak Mode 6)")
    print("  'q' : Quit unified test suite")
    print("=========================================================")
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1) # Mirror
        
        # HUD Panel at the bottom
        hud_height = 50
        cv2.rectangle(img, (0, h - hud_height), (w, h), (20, 20, 20), cv2.FILLED)
        cv2.line(img, (0, h - hud_height), (w, h - hud_height), (120, 120, 120), 2)
        
        modes_hud = "[1]Track [2]Draw [3]Write [4]Gesture [5]Count [6]Cloak [7]Clock [8]Blur [9]Mesh [0]Keybd"
        cv2.putText(img, modes_hud, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        active_modes = [
            "Virtual Keyboard", "Hand Tracking", "Virtual Drawing", "Air Writing", 
            "Gesture Control", "Finger Counter", "Invisibility Cloak", 
            "Clock & Stopwatch", "Selfie Segmentation", "Cyberpunk Face Mesh"
        ]
        cv2.putText(img, f"Active: {active_modes[mode]}", (w - 280, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Performance optimization: only run hand/face/selfie processing if required by active mode
        hands = []
        if mode in [1, 2, 3, 4, 5, 7, 0]:
            hands = detector.find_all_hands(img, draw=(mode not in [2, 3, 0]))
            
        # Clean up prev_points for drawing / writing
        active_hand_types = [hand["type"] for hand in hands]
        for h_type in list(prev_points.keys()):
            if h_type not in active_hand_types:
                prev_points[h_type] = (0, 0)
                
        # --- Mode 1: Hand Tracking & Finger Status ---
        if mode == 1:
            y_offset = 65
            if len(hands) > 0:
                cv2.rectangle(img, (15, 20), (230, 250), (45, 45, 45), cv2.FILLED)
                cv2.rectangle(img, (15, 20), (230, 250), (120, 120, 120), 2)
                cv2.putText(img, "FINGER STATES", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                for hand in hands:
                    h_type = hand["type"]
                    lmList = hand["lmList"]
                    fingers = detector.fingers_up(lmList, h_type)
                    
                    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                    text_str = f"{h_type} Hand: "
                    for i, name in enumerate(finger_names):
                        state = "U" if fingers[i] == 1 else "D"
                        text_str += f"{name[0]}:{state} "
                    
                    color = (0, 255, 0) if h_type == "Right" else (0, 255, 255)
                    cv2.putText(img, text_str, (w - 280 if h_type == "Left" else 20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw States card on left
                    if h_type == "Right":
                        for i, name in enumerate(finger_names):
                            state = "UP" if fingers[i] == 1 else "DOWN"
                            f_color = (0, 255, 0) if fingers[i] == 1 else (0, 0, 255)
                            cv2.putText(img, f"{name}: {state}", (35, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, f_color, 2)
                            y_offset += 35
            else:
                cv2.putText(img, "Show Hands to Track", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        # --- Mode 2: Virtual Drawing ---
        elif mode == 2:
            ui_height = 80
            num_colors = len(colors)
            color_width = w // num_colors
            
            for i, color in enumerate(colors):
                cv2.rectangle(img, (i * color_width, 0), ((i + 1) * color_width, ui_height), color, cv2.FILLED)
                if color == (0, 0, 0):
                    cv2.putText(img, "Eraser", (i * color_width + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if i == colorIndex:
                    cv2.rectangle(img, (i * color_width, 0), ((i + 1) * color_width, ui_height), (0, 0, 0), 4)
            
            for hand in hands:
                h_type = hand["type"]
                lmList = hand["lmList"]
                fingers = detector.fingers_up(lmList, h_type)
                
                if len(lmList) != 0:
                    x1, y1 = lmList[8][1:]
                    x2, y2 = lmList[12][1:]
                    
                    # Selection Mode (2 fingers up)
                    if len(fingers) >= 3 and fingers[1] and fingers[2]:
                        prev_points[h_type] = (0, 0)
                        if y1 < ui_height and (time.time() - last_selection_time > cooldown_delay):
                            colorIndex = x1 // color_width
                            if colorIndex >= num_colors:
                                colorIndex = num_colors - 1
                            drawColor = colors[colorIndex]
                            last_selection_time = time.time()
                        cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
                    
                    # Drawing Mode (1 finger up)
                    elif len(fingers) >= 2 and fingers[1] and not fingers[2]:
                        thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
                        xp, yp = prev_points[h_type]
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        
                        x_smooth = int(alpha * x1 + (1 - alpha) * xp)
                        y_smooth = int(alpha * y1 + (1 - alpha) * yp)
                        
                        cv2.circle(img, (x_smooth, y_smooth), thickness, drawColor, cv2.FILLED)
                        cv2.line(imgCanvas, (xp, yp), (x_smooth, y_smooth), drawColor, thickness)
                        prev_points[h_type] = (x_smooth, y_smooth)
                    else:
                        prev_points[h_type] = (0, 0)
                else:
                    prev_points[h_type] = (0, 0)
                    
            # Combine canvas
            imgCanvasGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgCanvasGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)
            
        # --- Mode 3: Air Writing ---
        elif mode == 3:
            cv2.putText(img, "Air Writing Mode (Pink Ink)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(img, "Eraser: 5 Fingers | Clear: Fist 1.5s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            fist_detected = False
            for hand in hands:
                h_type = hand["type"]
                lmList = hand["lmList"]
                fingers = detector.fingers_up(lmList, h_type)
                
                if len(lmList) != 0:
                    x1, y1 = lmList[8][1:]
                    
                    # 1. Closed Fist Gesture -> Clear Screen
                    if fingers.count(1) == 0:
                        fist_detected = True
                        prev_points[h_type] = (0, 0)
                    # 2. Eraser Gesture -> Erase
                    elif fingers.count(1) == 5:
                        prev_points[h_type] = (0, 0)
                        cv2.circle(img, (x1, y1), eraserThickness, (0, 0, 0), cv2.FILLED)
                        cv2.circle(imgAirCanvas, (x1, y1), eraserThickness, (0, 0, 0), cv2.FILLED)
                        cv2.putText(img, "ERASING", (x1 - 30, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # 3. Selection / Pause Mode
                    elif len(fingers) >= 3 and fingers[1] and fingers[2]:
                        prev_points[h_type] = (0, 0)
                        cv2.putText(img, "PAUSED", (x1 - 30, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # 4. Writing Mode
                    elif len(fingers) >= 2 and fingers[1] and not fingers[2]:
                        xp, yp = prev_points[h_type]
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        
                        x_smooth = int(alpha * x1 + (1 - alpha) * xp)
                        y_smooth = int(alpha * y1 + (1 - alpha) * yp)
                        
                        cv2.circle(img, (x_smooth, y_smooth), 8, airColor, cv2.FILLED)
                        cv2.line(imgAirCanvas, (xp, yp), (x_smooth, y_smooth), airColor, 8)
                        prev_points[h_type] = (x_smooth, y_smooth)
                    else:
                        prev_points[h_type] = (0, 0)
                else:
                    prev_points[h_type] = (0, 0)
                    
            if fist_detected and len(hands) > 0:
                if fist_start_time is None:
                    fist_start_time = time.time()
                elapsed = time.time() - fist_start_time
                remaining = max(0.0, clear_delay - elapsed)
                
                if elapsed >= clear_delay:
                    imgAirCanvas = np.zeros((h, w, 3), np.uint8)
                    fist_start_time = None
                    print("Air Canvas Cleared!")
                else:
                    bar_w = int(200 * (elapsed / clear_delay))
                    cv2.rectangle(img, (w // 2 - 100, h // 2 - 20), (w // 2 + 100, h // 2 + 10), (50, 50, 50), cv2.FILLED)
                    cv2.rectangle(img, (w // 2 - 100, h // 2 - 20), (w // 2 - 100 + bar_w, h // 2 + 10), (0, 0, 255), cv2.FILLED)
                    cv2.rectangle(img, (w // 2 - 100, h // 2 - 20), (w // 2 + 100, h // 2 + 10), (255, 255, 255), 2)
                    cv2.putText(img, f"Hold Fist to Clear: {remaining:.1f}s", (w // 2 - 140, h // 2 - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                fist_start_time = None
                
            # Combine canvas
            imgAirGray = cv2.cvtColor(imgAirCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgAirGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgAirCanvas)
            
        # --- Mode 4: Gesture Control ---
        elif mode == 4:
            for hand in hands:
                h_type = hand["type"]
                lmList = hand["lmList"]
                fingers = detector.fingers_up(lmList, h_type)
                
                if len(lmList) != 0:
                    # Closed Fist
                    if fingers.count(1) == 0:
                        cv2.putText(img, f"{h_type}: Pause/Stop Mode", (20, 60 if h_type == "Right" else 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Index only or Index+Thumb
                    elif fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
                        length, img, lineInfo = detector.find_distance(4, 8, img, lmList=lmList)
                        if length < 35: # Pinch
                            cv2.putText(img, f"{h_type}: Click/Zoom Pinch", (20, 60 if h_type == "Right" else 110), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                        else:
                            cv2.putText(img, f"{h_type}: Cursor Mode", (20, 60 if h_type == "Right" else 110), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Distance Slider
                            volBar = np.interp(length, [35, 220], [400, 150])
                            percent = np.interp(length, [35, 220], [0, 100])
                            x_offset = 60 if h_type == "Right" else w - 90
                            cv2.rectangle(img, (x_offset, 150), (x_offset + 30, 400), (0, 255, 0), 3)
                            cv2.rectangle(img, (x_offset, int(volBar)), (x_offset + 30, 400), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, f"{int(percent)}%", (x_offset - 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Index + Middle
                    elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                        cv2.putText(img, f"{h_type}: Drawing Mode", (20, 60 if h_type == "Right" else 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    else:
                        cv2.putText(img, f"{h_type}: Gesture Active", (20, 60 if h_type == "Right" else 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
        # --- Mode 5: Finger Counter ---
        elif mode == 5:
            total_sum = 0
            y_offset = 60
            for hand in hands:
                h_type = hand["type"]
                lmList = hand["lmList"]
                fingers = detector.fingers_up(lmList, h_type)
                count = fingers.count(1)
                total_sum += count
                
                cv2.putText(img, f"{h_type} Hand: {count} fingers", (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 40
                
            if len(hands) > 0:
                cv2.rectangle(img, (w - 240, 20), (w - 20, 200), (50, 150, 50), cv2.FILLED)
                cv2.rectangle(img, (w - 240, 20), (w - 20, 200), (255, 255, 255), 3)
                cv2.putText(img, "TOTAL", (w - 175, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, str(total_sum), (w - 165, 170), cv2.FONT_HERSHEY_PLAIN, 8, (255, 255, 255), 15)
            else:
                cv2.putText(img, "Show Hands to Count", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        # --- Mode 6: Invisibility Cloak ---
        elif mode == 6:
            if cloak_background is None:
                # Capture Background sequence
                if cloak_capture_start == 0:
                    cloak_capture_start = time.time()
                
                elapsed = time.time() - cloak_capture_start
                if elapsed < 2.0:
                    cv2.putText(img, "CAPTURING BACKGROUND...", (w // 2 - 250, h // 2 - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(img, "Please stay OUT of the camera frame!", (w // 2 - 270, h // 2 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                else:
                    cloak_background = img.copy()
                    cloak_capture_start = 0
                    print("Background Captured successfully in Unified Suite!")
            else:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower_red1 = np.array([0, 120, 70])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 120, 70])
                upper_red2 = np.array([180, 255, 255])
                
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = mask1 + mask2
                
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
                
                mask_inv = cv2.bitwise_not(mask)
                res1 = cv2.bitwise_and(cloak_background, cloak_background, mask=mask)
                res2 = cv2.bitwise_and(img, img, mask=mask_inv)
                img = cv2.addWeighted(res1, 1, res2, 1, 0)
                
                cv2.putText(img, "Invisibility Cloak: RED Color Substitute", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img, "Press 'b' to recapture static background", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
        # --- Mode 7: Clock & Gestures Stopwatch ---
        elif mode == 7:
            current_time = datetime.now().strftime("%I:%M:%S %p")
            date_str = datetime.now().strftime("%A, %B %d, %Y")
            
            # Draw Time Card
            cv2.rectangle(img, (w // 2 - 250, 20), (w // 2 + 250, 110), (35, 35, 35), cv2.FILLED)
            cv2.rectangle(img, (w // 2 - 250, 20), (w // 2 + 250, 110), (120, 120, 120), 2)
            cv2.putText(img, current_time, (w // 2 - 160, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            cv2.putText(img, date_str, (w // 2 - 140, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            stopwatch_text = "Stopwatch: Ready"
            if len(hands) > 0:
                hand = hands[0]
                fingers = detector.fingers_up(hand["lmList"], hand["type"])
                count = fingers.count(1)
                
                # Start / Resume (Index up)
                if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
                    if not stopwatch_running:
                        stopwatch_start_time = time.time() - stopwatch_elapsed_time
                        stopwatch_running = True
                    stopwatch_text = "Stopwatch: RUNNING"
                # Pause (Index + Middle up)
                elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                    if stopwatch_running:
                        stopwatch_elapsed_time = time.time() - stopwatch_start_time
                        stopwatch_running = False
                    stopwatch_text = "Stopwatch: PAUSED"
                # Reset (Fist)
                elif count == 0:
                    stopwatch_running = False
                    stopwatch_elapsed_time = 0
                    stopwatch_text = "Stopwatch: RESET"
                    
            disp_elapsed = stopwatch_elapsed_time
            if stopwatch_running:
                disp_elapsed = time.time() - stopwatch_start_time
                
            mins = int(disp_elapsed // 60)
            secs = int(disp_elapsed % 60)
            millis = int((disp_elapsed % 1) * 10)
            sw_str = f"{mins:02d}:{secs:02d}.{millis:1d}"
            
            # Stopwatch Card (top right)
            cv2.rectangle(img, (w - 320, 20), (w - 20, 150), (45, 45, 45), cv2.FILLED)
            cv2.rectangle(img, (w - 320, 20), (w - 20, 150), (0, 255, 0) if stopwatch_running else (0, 0, 255), 2)
            cv2.putText(img, "STOPWATCH", (w - 230, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, sw_str, (w - 250, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(img, stopwatch_text, (w - 300, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Legend (top left)
            cv2.rectangle(img, (20, 20), (300, 150), (40, 40, 40), cv2.FILLED)
            cv2.rectangle(img, (20, 20), (300, 150), (100, 100, 100), 2)
            cv2.putText(img, "STOPWATCH GESTURES", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(img, "- Index Up: START", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img, "- Index+Middle: PAUSE", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)
            cv2.putText(img, "- Closed Fist: RESET", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # --- Mode 8: Selfie Segmentation (Background Blur) ---
        elif mode == 8:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = selfie.process(imgRGB)
            mask = results.segmentation_mask
            condition = np.stack((mask,) * 3, axis=-1) > 0.5
            blurred_bg = cv2.GaussianBlur(img, (55, 55), 0)
            img = np.where(condition, img, blurred_bg)
            
            cv2.putText(img, "Portrait Mode (Translucent Background Blur)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        # --- Mode 9: Face Mesh Cyber Mask ---
        elif mode == 9:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(imgRGB)
            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        image=img,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=draw_spec,
                        connection_drawing_spec=conn_spec
                    )
            cv2.putText(img, "Cyberpunk Face Mesh (Full 468 Grid)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
        # --- Mode 10 (0 Key): Virtual Keyboard ---
        elif mode == 0:
            if pinch_cooldown > 0:
                pinch_cooldown -= 1
                
            index_pos = None
            is_pinching = False
            
            if len(hands) > 0:
                hand = hands[0]
                lmList = hand["lmList"]
                if len(lmList) != 0:
                    index_pos = lmList[8][1:]
                    length, img, lineInfo = detector.find_distance(4, 8, img, draw=False, lmList=lmList)
                    if length < 35:
                        is_pinching = True
                        
            # Draw Keys
            for key in keyboard_keys:
                kx, ky = key.pos
                kw, kh = key.size
                
                is_hovering = False
                if index_pos:
                    ix, iy = index_pos
                    if kx < ix < kx + kw and ky < iy < ky + kh:
                        is_hovering = True
                        
                if is_hovering and is_pinching:
                    color = (0, 255, 0)
                    text_color = (255, 255, 255)
                    
                    if pinch_cooldown == 0:
                        if key.char == "SPACE":
                            typed_text += " "
                        elif key.char == "BACK":
                            typed_text = typed_text[:-1]
                        elif key.char == "CLEAR":
                            typed_text = ""
                        else:
                            typed_text += key.char
                        pinch_cooldown = 12
                elif is_hovering:
                    color = (0, 180, 255)
                    text_color = (255, 255, 255)
                else:
                    color = (50, 50, 50)
                    text_color = (200, 200, 200)
                    
                cv2.rectangle(img, key.pos, (kx + kw, ky + kh), color, cv2.FILLED)
                cv2.rectangle(img, key.pos, (kx + kw, ky + kh), (150, 150, 150), 2)
                
                f_scale = 0.8
                if len(key.char) > 1: f_scale = 0.55
                (tw, th), _ = cv2.getTextSize(key.char, cv2.FONT_HERSHEY_SIMPLEX, f_scale, 2)
                tx = kx + (kw - tw) // 2
                ty = ky + (kh + th) // 2
                cv2.putText(img, key.char, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, f_scale, text_color, 2)
                
            # Draw Typed text bar
            cv2.rectangle(img, (100, 500), (w - 100, 565), (30, 30, 30), cv2.FILLED)
            cv2.rectangle(img, (100, 500), (w - 100, 565), (255, 255, 255), 2)
            cv2.putText(img, typed_text + ("|" if (time.time() % 1.0 > 0.5) else ""), (120, 545), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(img, "Virtual Interactive Keyboard", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Hover to highlight | Pinch Thumb+Index to TYPE", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
        cv2.imshow("YasirVision - Unified Interactive Testing Suite", img)
        
        # Key controls
        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            break
        elif key_press == ord('1'):
            mode = 1
        elif key_press == ord('2'):
            mode = 2
        elif key_press == ord('3'):
            mode = 3
        elif key_press == ord('4'):
            mode = 4
        elif key_press == ord('5'):
            mode = 5
        elif key_press == ord('6'):
            mode = 6
        elif key_press == ord('7'):
            mode = 7
        elif key_press == ord('8'):
            mode = 8
        elif key_press == ord('9'):
            mode = 9
        elif key_press == ord('0'):
            mode = 0
        elif key_press == ord('c'):
            imgCanvas = np.zeros((h, w, 3), np.uint8)
            imgAirCanvas = np.zeros((h, w, 3), np.uint8)
            xp, yp = 0, 0
            xp_air, yp_air = 0, 0
            print("Canvas cleared in interactive loop!")
        elif key_press == ord('b'):
            cloak_background = None
            print("Re-triggering Background Capture...")
            
    selfie.close()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import time
import math
from datetime import datetime
from .core import HandDetector

def hand_tracking():
    """
    A simple plug-and-play function that opens the webcam and tracks all hands.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2, detectionCon=0.8)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture from webcam.")
            break
        img = cv2.flip(img, 1)
        hands = detector.find_all_hands(img, draw=True)
        
        # Display individual finger states on screen
        y_offset = 50
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
            cv2.putText(img, text_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 35
            
        cv2.imshow("YasirVision - Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def virtual_drawing_canvas():
    """
    Opens the webcam and creates a virtual drawing canvas.
    Use index finger to draw. Use two fingers (index + middle) to select colors.
    """
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 255, 255),# White
        (0, 0, 0)       # Eraser
    ]
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    detector = HandDetector(maxHands=2, detectionCon=0.8)
    
    colorIndex = 0
    drawColor = colors[colorIndex]
    brushThickness = 15
    eraserThickness = 50
    
    # Store previous points for each hand to enable smooth independent drawing
    prev_points = {"Left": (0, 0), "Right": (0, 0)}
    alpha = 0.35 # EMA smoothing factor
    
    success, img = cap.read()
    if not success:
        return
    h, w, c = img.shape
    imgCanvas = np.zeros((h, w, 3), np.uint8)
    
    # Selection Cooldown to avoid accidental trigger
    last_selection_time = 0
    cooldown_delay = 0.5 # seconds

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        hands = detector.find_all_hands(img, draw=True)
        
        # Draw UI (Color Palette)
        ui_height = 80
        num_colors = len(colors)
        color_width = w // num_colors
        
        for i, color in enumerate(colors):
            cv2.rectangle(img, (i * color_width, 0), ((i + 1) * color_width, ui_height), color, cv2.FILLED)
            if color == (0, 0, 0):
                cv2.putText(img, "Eraser", (i * color_width + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if i == colorIndex:
                cv2.rectangle(img, (i * color_width, 0), ((i + 1) * color_width, ui_height), (0, 0, 0), 4)

        active_hand_types = [hand["type"] for hand in hands]
        # Clean up prev_points for hands no longer visible
        for h_type in list(prev_points.keys()):
            if h_type not in active_hand_types:
                prev_points[h_type] = (0, 0)

        for hand in hands:
            h_type = hand["type"]
            lmList = hand["lmList"]
            fingers = detector.fingers_up(lmList, h_type)
            
            if len(lmList) != 0:
                x1, y1 = lmList[8][1:] # Index Finger
                x2, y2 = lmList[12][1:] # Middle Finger
                
                # Selection Mode - Two fingers are up
                if len(fingers) >= 3 and fingers[1] and fingers[2]:
                    prev_points[h_type] = (0, 0) # Reset drawing history
                    
                    # Prevent accidental selections when drawing close to the edge
                    if y1 < ui_height and (time.time() - last_selection_time > cooldown_delay):
                        colorIndex = x1 // color_width
                        if colorIndex >= num_colors:
                            colorIndex = num_colors - 1
                        drawColor = colors[colorIndex]
                        last_selection_time = time.time()
                        
                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
                
                # Drawing Mode - Index finger is up
                elif len(fingers) >= 2 and fingers[1] and not fingers[2]:
                    thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
                    
                    # Apply EMA coordinate smoothing
                    xp, yp = prev_points[h_type]
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    
                    x_smooth = int(alpha * x1 + (1 - alpha) * xp)
                    y_smooth = int(alpha * y1 + (1 - alpha) * yp)
                    
                    cv2.circle(img, (x_smooth, y_smooth), thickness, drawColor, cv2.FILLED)
                    
                    # Draw on canvas
                    cv2.line(imgCanvas, (xp, yp), (x_smooth, y_smooth), drawColor, thickness)
                    prev_points[h_type] = (x_smooth, y_smooth)
                else:
                    prev_points[h_type] = (0, 0)
            else:
                prev_points[h_type] = (0, 0)

        # Combine canvas with original feed
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        cv2.imshow("YasirVision - Virtual Drawing Canvas", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def air_write():
    """
    Similar to virtual drawing canvas, but a streamlined version specifically for writing.
    Includes a 5-finger Eraser gesture and a Fist-hold countdown to clear the screen!
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    detector = HandDetector(maxHands=2, detectionCon=0.8)
    
    success, img = cap.read()
    if not success:
        return
    h, w, c = img.shape
    imgCanvas = np.zeros((h, w, 3), np.uint8)
    
    # Writing states
    prev_points = {"Left": (0, 0), "Right": (0, 0)}
    alpha = 0.35
    drawColor = (255, 0, 255) # Pink ink
    eraserThickness = 45
    
    # Fist Hold Clear States
    fist_start_time = None
    clear_delay = 1.5 # seconds required to clear
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        hands = detector.find_all_hands(img, draw=True)
        
        active_hand_types = [hand["type"] for hand in hands]
        for h_type in list(prev_points.keys()):
            if h_type not in active_hand_types:
                prev_points[h_type] = (0, 0)
                
        fist_detected = False
        
        for hand in hands:
            h_type = hand["type"]
            lmList = hand["lmList"]
            fingers = detector.fingers_up(lmList, h_type)
            
            if len(lmList) != 0:
                x1, y1 = lmList[8][1:] # Index tip
                
                # 1. Closed Fist Gesture (0 fingers up) -> Clear screen
                if fingers.count(1) == 0:
                    fist_detected = True
                    prev_points[h_type] = (0, 0)
                    
                # 2. Eraser Gesture (5 fingers up) -> Erase canvas around index tip
                elif fingers.count(1) == 5:
                    prev_points[h_type] = (0, 0)
                    cv2.circle(img, (x1, y1), eraserThickness, (0, 0, 0), cv2.FILLED)
                    cv2.circle(imgCanvas, (x1, y1), eraserThickness, (0, 0, 0), cv2.FILLED)
                    cv2.putText(img, "ERASING", (x1 - 30, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 3. Selection / Pause Mode (2 fingers up)
                elif len(fingers) >= 3 and fingers[1] and fingers[2]:
                    prev_points[h_type] = (0, 0)
                    cv2.putText(img, "PAUSED", (x1 - 30, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                # 4. Writing Mode (Index finger only)
                elif len(fingers) >= 2 and fingers[1] and not fingers[2]:
                    xp, yp = prev_points[h_type]
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    
                    x_smooth = int(alpha * x1 + (1 - alpha) * xp)
                    y_smooth = int(alpha * y1 + (1 - alpha) * yp)
                    
                    cv2.circle(img, (x_smooth, y_smooth), 8, drawColor, cv2.FILLED)
                    cv2.line(imgCanvas, (xp, yp), (x_smooth, y_smooth), drawColor, 8)
                    prev_points[h_type] = (x_smooth, y_smooth)
                else:
                    prev_points[h_type] = (0, 0)
            else:
                prev_points[h_type] = (0, 0)
                
        # Handle Fist-Hold Clear Countdown Logic
        if fist_detected and len(hands) > 0:
            if fist_start_time is None:
                fist_start_time = time.time()
            
            elapsed = time.time() - fist_start_time
            remaining = max(0.0, clear_delay - elapsed)
            
            if elapsed >= clear_delay:
                imgCanvas = np.zeros((h, w, 3), np.uint8)
                fist_start_time = None
                print("Canvas Cleared by Fist Gesture!")
            else:
                bar_w = int(200 * (elapsed / clear_delay))
                cv2.rectangle(img, (w // 2 - 100, h // 2 - 20), (w // 2 + 100, h // 2 + 10), (50, 50, 50), cv2.FILLED)
                cv2.rectangle(img, (w // 2 - 100, h // 2 - 20), (w // 2 - 100 + bar_w, h // 2 + 10), (0, 0, 255), cv2.FILLED)
                cv2.rectangle(img, (w // 2 - 100, h // 2 - 20), (w // 2 + 100, h // 2 + 10), (255, 255, 255), 2)
                cv2.putText(img, f"Hold Fist to Clear: {remaining:.1f}s", (w // 2 - 140, h // 2 - 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            fist_start_time = None

        # Combine canvas with original feed
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        cv2.putText(img, "Air Writing Mode", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "Eraser: 5 Fingers | Clear: Fist 1.5s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("YasirVision - Air Writing", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def gesture_control():
    """
    Demo of modular gesture control system.
    Recognizes:
    - Index up: Cursor mode
    - Index + Middle up: Drawing mode
    - Closed fist: Pause/stop
    - Pinch (Thumb + Index close): Zoom simulation
    - Thumb-Index distance: Volume/brightness simulation
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(maxHands=2, detectionCon=0.8)

    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        hands = detector.find_all_hands(img, draw=True)
        
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
                        
                        # Distance Slider simulation
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

        cv2.imshow("YasirVision - Gesture Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def finger_counter():
    """
    Counts and displays the number of fingers held up on both hands.
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(maxHands=2, detectionCon=0.8)
    
    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        hands = detector.find_all_hands(img, draw=True)
        
        total_sum = 0
        y_offset = 60
        
        for hand in hands:
            h_type = hand["type"]
            lmList = hand["lmList"]
            fingers = detector.fingers_up(lmList, h_type)
            count = fingers.count(1)
            total_sum += count
            
            # Show count for each hand
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
            
        cv2.imshow("YasirVision - Finger Counter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def invisibility_cloak():
    """
    Simulates an invisibility cloak by substituting a specific color (default: Red) 
    with a pre-captured static background.
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    print("Invisibility Cloak: Please stay OUT of the frame for 2 seconds to capture the background.")
    
    background = None
    capture_start = time.time()
    
    # 2 seconds background capture
    while time.time() - capture_start < 2.0:
        success, frame = cap.read()
        if success:
            background = cv2.flip(frame, 1)
            cv2.putText(background, "CAPTURING BACKGROUND...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("YasirVision - Invisibility Cloak", background)
            cv2.waitKey(1)
            
    # Clean the background text
    success, frame = cap.read()
    if success:
        background = cv2.flip(frame, 1)
        
    print("Background Captured! You can step back in with a RED cloth/object.")

    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Ranges for detecting RED color
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Refine mask (Morphological Operations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        
        # Create inverted mask (for non-red parts)
        mask_inv = cv2.bitwise_not(mask)
        
        # Segment red color out (replace with background)
        res1 = cv2.bitwise_and(background, background, mask=mask)
        
        # Keep non-red color parts of feed
        res2 = cv2.bitwise_and(img, img, mask=mask_inv)
        
        # Combined final frame
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
        
        cv2.putText(final_output, "Invisibility Cloak Mode (Red Color)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("YasirVision - Invisibility Cloak", final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def virtual_clock_stopwatch():
    """
    Displays an elegant digital clock on screen and provides a fully interactive 
    gesture-controlled stopwatch (1 finger=Start, 2 fingers=Pause, Fist=Reset).
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(maxHands=1, detectionCon=0.8)
    
    # Stopwatch states
    running = False
    start_time = 0
    elapsed_time = 0
    
    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        hands = detector.find_all_hands(img, draw=True)
        
        # 1. Overlay Digital Clock (at top center)
        current_time = datetime.now().strftime("%I:%M:%S %p")
        date_str = datetime.now().strftime("%A, %B %d, %Y")
        
        cv2.rectangle(img, (w // 2 - 250, 20), (w // 2 + 250, 110), (35, 35, 35), cv2.FILLED)
        cv2.rectangle(img, (w // 2 - 250, 20), (w // 2 + 250, 110), (120, 120, 120), 2)
        cv2.putText(img, current_time, (w // 2 - 160, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.putText(img, date_str, (w // 2 - 140, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Handle Gestures for Stopwatch
        gesture_text = "Stopwatch: Ready"
        if len(hands) > 0:
            hand = hands[0]
            fingers = detector.fingers_up(hand["lmList"], hand["type"])
            count = fingers.count(1)
            
            # Start/Resume (Index Finger Up: [0,1,0,0,0] or [1,1,0,0,0])
            if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
                if not running:
                    start_time = time.time() - elapsed_time
                    running = True
                gesture_text = "Stopwatch: RUNNING (Index Up)"
            # Pause (Index + Middle Up: [0,1,1,0,0] or [1,1,1,0,0])
            elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                if running:
                    elapsed_time = time.time() - start_time
                    running = False
                gesture_text = "Stopwatch: PAUSED (2 Fingers Up)"
            # Reset (Fist: count = 0)
            elif count == 0:
                running = False
                elapsed_time = 0
                gesture_text = "Stopwatch: RESET (Fist)"
        
        # Update elapsed time if running
        display_elapsed = elapsed_time
        if running:
            display_elapsed = time.time() - start_time
            
        # Draw Stopwatch Panel (at top right)
        mins = int(display_elapsed // 60)
        secs = int(display_elapsed % 60)
        millis = int((display_elapsed % 1) * 10)
        stopwatch_str = f"{mins:02d}:{secs:02d}.{millis:1d}"
        
        cv2.rectangle(img, (w - 320, 20), (w - 20, 150), (45, 45, 45), cv2.FILLED)
        cv2.rectangle(img, (w - 320, 20), (w - 20, 150), (0, 255, 0) if running else (0, 0, 255), 2)
        cv2.putText(img, "STOPWATCH", (w - 230, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, stopwatch_str, (w - 250, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(img, gesture_text, (w - 300, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw Gesture Legend on left
        cv2.rectangle(img, (20, 20), (320, 150), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(img, (20, 20), (320, 150), (100, 100, 100), 2)
        cv2.putText(img, "GESTURE LEGEND", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "- Index Up: START", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, "- Index+Middle: PAUSE", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)
        cv2.putText(img, "- Closed Fist: RESET", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("YasirVision - Virtual Clock & Stopwatch", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def selfie_segmentation():
    """
    Applies Gaussian Blur to the background using MediaPipe Selfie Segmentation.
    """
    import mediapipe as mp
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    mp_selfie = mp.solutions.selfie_segmentation
    selfie = mp_selfie.SelfieSegmentation(model_selection=1) # 1 for landscape, 0 for general

    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        # RGB Conversion
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = selfie.process(imgRGB)
        
        mask = results.segmentation_mask
        
        # Threshold mask to make it solid binary
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        
        # Create Blurred Background
        blurred_bg = cv2.GaussianBlur(img, (55, 55), 0)
        
        # Combine images based on segmentation mask
        output_image = np.where(condition, img, blurred_bg)
        
        cv2.putText(output_image, "Selfie Segmentation (Background Blur)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("YasirVision - Selfie Segmentation", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    selfie.close()
    cap.release()
    cv2.destroyAllWindows()

def face_mesh_detector():
    """
    Detects and draws a beautiful cybernetic 3D face mesh using MediaPipe Face Mesh.
    """
    import mediapipe as mp
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    # Create stylish glowing cyber drawing specs
    draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))
    conn_spec = mp_draw.DrawingSpec(thickness=1, color=(0, 255, 0))

    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
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
                
        cv2.putText(img, "Face Mesh Mode (Cyberpunk Net)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("YasirVision - Face Mesh", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

def virtual_keyboard():
    """
    Draws a futuristic virtual QWERTY keyboard on screen.
    Hover index finger over keys and pinch (Thumb+Index) to type!
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(maxHands=1, detectionCon=0.8)
    
    # Keyboard layout
    keys = [
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
        ["SPACE", "BACK", "CLEAR"]
    ]
    
    # Class representation of Keys
    class Key:
        def __init__(self, char, pos, size=(80, 80)):
            self.char = char
            self.pos = pos
            self.size = size # (width, height)
            
    # Initialize all keys
    keyboard_keys = []
    y_start = 150
    for row in keys:
        x_start = 100
        for char in row:
            if char in ["SPACE", "BACK", "CLEAR"]:
                width = 250 if char == "SPACE" else 150
                keyboard_keys.append(Key(char, (x_start, y_start), (width, 70)))
                x_start += width + 20
            else:
                keyboard_keys.append(Key(char, (x_start, y_start), (80, 70)))
                x_start += 100
        y_start += 90
        
    typed_text = ""
    pinch_cooldown = 0
    
    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        hands = detector.find_all_hands(img, draw=True)
        
        # Decrement cooldown
        if pinch_cooldown > 0:
            pinch_cooldown -= 1
            
        index_pos = None
        is_pinching = False
        
        if len(hands) > 0:
            hand = hands[0]
            lmList = hand["lmList"]
            
            if len(lmList) != 0:
                # Check for pinch
                length, img, lineInfo = detector.find_distance(4, 8, img, draw=False, lmList=lmList)
                if length < 35:
                    is_pinching = True
                    
        # Draw Keyboard
        for key in keyboard_keys:
            kx, ky = key.pos
            kw, kh = key.size
            
            # Check if index finger is hovering
            is_hovering = False
            if index_pos:
                ix, iy = index_pos
                if kx < ix < kx + kw and ky < iy < ky + kh:
                    is_hovering = True
                    
            # Color states
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
                    pinch_cooldown = 15 # frames of cooldown
            elif is_hovering:
                color = (0, 180, 255)
                text_color = (255, 255, 255)
            else:
                color = (55, 55, 55)
                text_color = (200, 200, 200)
                
            cv2.rectangle(img, key.pos, (kx + kw, ky + kh), color, cv2.FILLED)
            cv2.rectangle(img, key.pos, (kx + kw, ky + kh), (150, 150, 150), 2)
            
            font_scale = 0.8
            if len(key.char) > 1: font_scale = 0.6
            (tw, th), _ = cv2.getTextSize(key.char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            tx = kx + (kw - tw) // 2
            ty = ky + (kh + th) // 2
            cv2.putText(img, key.char, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
            
        # Draw Typed Text Bar
        cv2.rectangle(img, (100, 520), (w - 100, 590), (30, 30, 30), cv2.FILLED)
        cv2.rectangle(img, (100, 520), (w - 100, 590), (255, 255, 255), 2)
        cv2.putText(img, typed_text + ("|" if (time.time() % 1.0 > 0.5) else ""), (120, 565), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(img, "Virtual Interactive Keyboard", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "Hover to highlight | Pinch Thumb+Index to TYPE", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.imshow("YasirVision - Virtual Keyboard", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

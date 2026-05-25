import cv2
from yasirvision import HandDetector

def main():
    # 1. Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # 2. Initialize the YasirVision HandDetector
    detector = HandDetector(detectionCon=0.8)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture from webcam.")
            break

        # 3. Find hands in the image
        img = detector.find_hands(img)
        
        # 4. Find landmarks of the first hand
        lmList = detector.find_position(img, draw=False)
        
        if len(lmList) != 0:
            # Check which fingers are up
            fingers = detector.fingers_up()
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            y_offset = 50
            for i, name in enumerate(finger_names):
                if i < len(fingers):
                    state = "UP" if fingers[i] == 1 else "DOWN"
                    color = (0, 255, 0) if fingers[i] == 1 else (0, 0, 255)
                    cv2.putText(img, f"{name}: {state}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += 35

        # 5. Display the result
        cv2.imshow("YasirVision - Hand Tracking Example", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

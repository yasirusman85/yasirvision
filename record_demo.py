import cv2
import yasirvision as yv

def main():
    print("Recording YasirVision Demo...")
    cap = cv2.VideoCapture(0)
    
    # Get standard dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        width, height = 640, 480
        

    
    detector = yv.HandDetector(detectionCon=0.8)
    
    print("Press 'q' to stop recording.")
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = detector.find_hands(img)
        img = cv2.flip(img, 1) # Mirror
        
        lmList = detector.find_position(img, draw=False)
        if len(lmList) != 0:
            fingers = detector.fingers_up()
            cv2.putText(img, f"Fingers: {fingers.count(1)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        cv2.imshow('Recording Demo - YasirVision', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo finished successfully!")

if __name__ == "__main__":
    main()

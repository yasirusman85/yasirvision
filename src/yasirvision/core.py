import cv2
import mediapipe as mp
import math

class HandDetector:
    """
    A simple wrapper for MediaPipe's hand tracking, optimized for readability and ease of use.
    """
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None
        self.lmList = []

    def find_hands(self, img, draw=True):
        """
        Detects hands in an image and optionally draws the landmarks.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNo=0, draw=True):
        """
        Returns a list of landmarks for a specific hand.
        """
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def find_all_hands(self, img, draw=True):
        """
        Detects all hands in the image.
        Returns a list of dictionaries, each representing a hand:
        {
            "lmList": [[id, x, y], ...],
            "bbox": (xmin, ymin, w, h),
            "center": (cx, cy),
            "type": "Left" or "Right"
        }
        """
        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        all_hands = []
        
        if self.results.multi_hand_landmarks:
            for handNo, handLms in enumerate(self.results.multi_hand_landmarks):
                # Determine handedness
                hand_type = "Right"
                if self.results.multi_handedness:
                    label = self.results.multi_handedness[handNo].classification[0].label
                    # MediaPipe classifies from webcam perspective (which is inverted)
                    hand_type = label
                
                lmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    xList.append(cx)
                    yList.append(cy)
                
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
                
                all_hands.append({
                    "lmList": lmList,
                    "bbox": bbox,
                    "center": (cx, cy),
                    "type": hand_type
                })
                
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    # Draw a high-tech corner box instead of a simple rectangle
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
                    cv2.putText(img, hand_type, (xmin - 20, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
        return all_hands

    def fingers_up(self, lmList=None, hand_type="Right"):
        """
        Checks which fingers are up.
        Supports automatic Left/Right thumb mapping.
        Returns a list of 5 integers (1 for up, 0 for down).
        """
        if lmList is None:
            lmList = self.lmList
            
        fingers = []
        if not lmList or len(lmList) < 21:
            return fingers
            
        # Thumb: We compare the tip (4) to the IP joint (3) or MCP joint (2).
        # On a Right hand (webcam view, mirrored):
        # A raised thumb pointing left means x4 < x3.
        # However, MediaPipe's Left/Right labels are based on standard skeletal coordinates.
        # Let's adjust based on Handedness:
        # Standard right-hand thumb points left when open. Standard left-hand thumb points right.
        if hand_type == "Right":
            if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else: # Left hand
            if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            # If the tip (id) is higher (smaller Y) than the PIP joint (id-2)
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3, lmList=None):
        """
        Finds distance between two landmarks based on their index numbers.
        Supports boundary protection and custom lmList.
        """
        if lmList is None:
            lmList = self.lmList
            
        if not lmList or len(lmList) <= max(p1, p2):
            return 0, img, [0, 0, 0, 0, 0, 0]
            
        x1, y1 = lmList[p1][1:]
        x2, y2 = lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

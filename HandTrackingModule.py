import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, max_hands = 2, mod_complex= 1, detection_confidence = 0.5, tracking_confidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.mod_complex = mod_complex
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.mod_complex, self.detection_confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_pos(self, img, hand_num=0, draw=True):

        landmark_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]

            for id, lm in enumerate(my_hand.landmark):
                height, width, c = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return landmark_list


def main():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        landmark_list = detector.find_pos(img)
        if len(landmark_list) != 0:
            print(landmark_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Hands", img)
        cv2.waitKey(1)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Hands", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
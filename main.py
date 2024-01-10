from Classes.CameraCapture import CameraCapture
from Classes.CardDetector import CardDetector
import cv2

camera = CameraCapture()
detector = CardDetector()
camera.open_camera()

try:
    while True:
        frame = camera.get_frame()

        cards = detector.detect_cards(frame)
        for card in cards:
            cv2.drawContours(frame, [card], -1, (0, 255, 0), 2)

        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Kamera', grayImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release_camera()
    cv2.destroyAllWindows()

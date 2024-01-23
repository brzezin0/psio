from Classes.CameraCapture import CameraCapture
from Classes.CardDetector import CardDetector
import cv2

# camera = CameraCapture()
# detector = CardDetector()
# camera.open_camera()
#
# try:
#     while True:
#         frame = camera.get_frame()
#         # adjust threshold values
#         _, threshold = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)
#         # adjust zoom
#         # focus  = frame[x1:x2,y1:y2]
#
#         # cards = detector.detect_cards(frame)
#         # for card in cards:
#         #     cv2.drawContours(frame, [card], -1, (0, 255, 0), 2)
#
#         cv2.imshow('Kamera', threshold)
#
#         key = cv2.waitKey(1)
#         if key == 27:
#             break
#
# finally:
#     camera.release_camera()
#     cv2.destroyAllWindows()

similarity_threshold = 0.99999999
detector = CardDetector()

is_card, card_image = detector.detect_playing_card("test1.jpg", "test.jpg", similarity_threshold)
if is_card:
    print("Wykryto kartę do gry.")
else:
    print("Nie wykryto karty do gry.")

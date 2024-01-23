import cv2
import numpy as np

CARD_SIGNATURE_HEIGHT = 145
CARD_SIGNATURE_WIDTH = 52


def preprocess_image(image):
    # Apply GaussianBlur to reduce noise and help edge detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred


def find_cards(image_after_preprocess, original_image):
    # Find contours in the edged image
    edges = cv2.Canny(image_after_preprocess, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)

    # Iterate through the contours and filter out potential card shapes
    card_contours = []
    count = 0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # Card shapes typically have 4 corners
        if len(approx) == 4:
            card_contours.append(approx)
            count += 1
    print(count)
    # Extract the region of interest (ROI) for each card
    cards = []
    for card_contour in card_contours:
        x, y, w, h = cv2.boundingRect(card_contour)
        if w < 100 or h < 100:
            continue
        print("x: ", x, "y:", y, "w:", w, "h: ", h)

        # Sorting a list of corners based on the second element's value
        sorted_list = sorted(card_contour, key=lambda x: x[0][1])

        # Apply perspective transformation; Matching corners of image contours
        pts1 = np.float32(sorted_list)
        if (sorted_list[1][0][0] > sorted_list[0][0][0]):
            if (sorted_list[2][0][0] > sorted_list[3][0][0]):
                pts2 = np.float32([[0, 0], [528, 0], [528, 378], [0, 378]])
            else:
                pts2 = np.float32([[0, 0], [528, 0], [0, 378], [528, 378]])
        elif (sorted_list[2][0][0] > sorted_list[3][0][0]):
            pts2 = np.float32([[528, 0], [0, 0], [528, 378], [0, 378]])
        else:
            pts2 = np.float32([[528, 0], [0, 0], [0, 378], [528, 378]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_card = cv2.warpPerspective(original_image, matrix, (528, 378))

        # Rotating the image to be vertical
        rotated_image = cv2.transpose(transformed_card)
        rotated_image = cv2.flip(rotated_image, 1)

        cards.append(rotated_image)

    return cards, card_contours


def preprocess_card(card):
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_level = 155
    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
    return thresh


def match_card(card):
    pass


def calc_similarity_between_signatures(image_signature, pattern_signature):
    diff_img = cv2.absdiff(image_signature, pattern_signature)
    rank_diff = int(np.sum(diff_img) / 255)
    return rank_diff


def get_card_signature(card):
    part = card[0:CARD_SIGNATURE_HEIGHT, 0:CARD_SIGNATURE_WIDTH]
    part_thresh = preprocess_card(part)
    return part_thresh


def show_card_from_frame(frame):
    image1 = preprocess_image(frame)
    cards, card_contours = find_cards(image1, frame)

    for card, contour in zip(cards, card_contours):
        cv2.imshow("Card", card)
        cv2.waitKey(20)


#################   MAIN - wersja statyczna z pojedynczym zdjęciem ###########################################

# image_filename = "kier_as.jpg"
# pattern_signature_filename = "kier_dama.jpg"
#
# image = cv2.imread("./images/" + image_filename)
# image_after_preprocess = preprocess_image(image)
#
# resized_image = cv2.resize(image_after_preprocess, (960, 540))
# cv2.imshow("Image", resized_image)
# cv2.waitKey(0)
#
# pattern_signature = cv2.imread("./signatures/"+pattern_signature_filename)
# pattern_signature = cv2.cvtColor(pattern_signature, cv2.COLOR_BGR2GRAY)
#
# cards, card_contours = find_cards(image_after_preprocess, image)
#
# for card, contour in zip(cards, card_contours):
#     cv2.imshow("Card", card)
#
#     card_signature = get_card_signature(card)
#     cv2.imshow("Card Signature", card_signature)
#
#     cv2.imshow("Patern", pattern_signature)
#     cv2.waitKey(0)
#
#     diff_between_signatures = calc_similarity_between_signatures(card_signature,pattern_signature)
#     print(diff_between_signatures)
#
# #     cv2.imwrite("./signatures/"+image_filename, card_signature)
# #     cv2.imwrite("./patterns/"+pattern_signature_filename, card)
# 
#
# cv2.destroyAllWindows()


#################   MAIN - wersja z wideo ###########################################

video_path = "./images/wideo.mp4"
cap = cv2.VideoCapture(video_path)

# Sprawdź, czy plik wideo został prawidłowo otwarty
if not cap.isOpened():
    print("Błąd podczas otwierania pliku wideo.")
    exit()

# Ustaw nowe wymiary ramki (np. szerokość = 500, wysokość = 300)
frame_width = 500
frame_height = 300

frame_indx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Koniec pliku wideo.")
        break

    if frame_indx == 20:
        show_card_from_frame(frame)
        frame_indx = 0

    resized_frame = cv2.resize(frame, (frame_width, frame_height))

    cv2.imshow('Frame', resized_frame)
    frame_indx += 1

    if cv2.waitKey(50) & 0xFF == ord('q'):  # Czekaj 50 ms między klatkami
        break

cap.release()
cv2.destroyAllWindows()
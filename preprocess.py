import cv2
import numpy as np

CARD_SIGNATURE_HEIGHT = 145
CARD_SIGNATURE_WIDTH = 52


class Card:
    def __init__(self, image, signature):
        self.image = image
        self.signature = signature


def imread_patterns_and_signatures(filenames):
    result = {}
    pattern_folder_path = "./patterns/"
    signature_folder_path = "./signatures/"
    file_extension = ".jpg"

    for filename in filenames:
        pattern = cv2.imread(pattern_folder_path + filename + file_extension)
        signature = cv2.imread(signature_folder_path + filename + file_extension)
        signature = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
        card = Card(pattern, signature)
        result[filename] = card

    return result


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


def match_card(card_signature, cards_patern_and_signature, card_names):
    similarity = 99999  # im mniejsza wartosc tym karty podobne
    match_card_name = None

    for card_name in card_names:
        diff_between_signatures = calc_similarity_between_signatures(card_signature,
                                                                     cards_patern_and_signature[card_name].signature)
        if diff_between_signatures < similarity:
            similarity = diff_between_signatures
            match_card_name = card_name

    print(similarity)
    print(match_card_name)
    cv2.imshow("Card Signature", cards_patern_and_signature[match_card_name].signature)
    cv2.waitKey(0)
    print("\n")


def calc_similarity_between_signatures(image_signature, pattern_signature):
    diff_img = cv2.absdiff(image_signature, pattern_signature)
    rank_diff = int(np.sum(diff_img) / 255)
    return rank_diff


def get_card_signature(card):
    part = card[0:CARD_SIGNATURE_HEIGHT, 0:CARD_SIGNATURE_WIDTH]
    part_thresh = preprocess_card(part)
    return part_thresh


def show_card_from_frame(frame, cards_pattern_and_signature, filenames):
    image1 = preprocess_image(frame)
    cards, card_contours = find_cards(image1, frame)

    for card, contour in zip(cards, card_contours):
        cv2.imshow("Card", card)

        card_signature = get_card_signature(card)
        match_card(card_signature, cards_pattern_and_signature, filenames)

        cv2.waitKey(1000)


filenames = ["trefl_as", "trefl_krol", "trefl_dama", "trefl_walet", "trefl_10", "trefl_9", "trefl_8", "trefl_7",
             "trefl_6",
             "trefl_5", "trefl_4", "trefl_3", "trefl_2",
             "kier_as", "kier_krol", "kier_dama", "kier_walet", "kier_10", "kier_9", "kier_8", "kier_7",
             "karo_7"]

#################   MAIN - wersja statyczna z pojedynczym zdjęciem ###########################################

# cards_pattern_and_signature =  imread_patterns_and_signatures(filenames)

# image_filename = "trefl_as.jpg"

# image = cv2.imread(image_filename)
# image_after_preprocess = preprocess_image(image)

# resized_image = cv2.resize(image_after_preprocess, (960, 540))
# cv2.imshow("Image", resized_image)
# cv2.waitKey(0)

# cards, card_contours = find_cards(image_after_preprocess, image)

# for card, contour in zip(cards, card_contours):
#     cv2.imshow("Card", card)

#     card_signature = get_card_signature(card)
#     cv2.imshow("Card Signature", card_signature)

#     match_card(card_signature, cards_pattern_and_signature, filenames)
#     cv2.waitKey(0)

# #     cv2.imwrite("./figury/"+image_filename, card_signature)
# #     cv2.imwrite("./wzorce_karty/"+pattern_signature_filename, card)


# cv2.destroyAllWindows()


#################   MAIN - wersja z wideo ###########################################

cards_pattern_and_signature = imread_patterns_and_signatures(filenames)

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

    if frame_indx == 90:
        show_card_from_frame(frame, cards_pattern_and_signature, filenames)
        frame_indx = 0

    resized_frame = cv2.resize(frame, (frame_width, frame_height))

    cv2.imshow('Frame', resized_frame)
    frame_indx += 1

    if cv2.waitKey(50) & 0xFF == ord('q'):  # Czekaj 50 ms między klatkami
        break

cap.release()
cv2.destroyAllWindows()
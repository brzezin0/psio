import cv2
import numpy as np

CARD_SIGNATURE_HEIGHT = 145
CARD_SIGNATURE_WIDTH = 52
MIN_CARD_WIDTH = 400
MAX_CARD_WIDTH = 600
MIN_CARD_HEIGHT = 280
MAX_CARD_HEIGHT = 400

CARDS_NAME = ["trefl_as", "trefl_krol", "trefl_dama", "trefl_walet", "trefl_10", "trefl_9", "trefl_8", "trefl_7",
             "trefl_6",
             "trefl_5", "trefl_4", "trefl_3", "trefl_2",
             "kier_as", "kier_krol", "kier_dama", "kier_walet", "kier_10", "kier_9", "kier_8", "kier_7", "kier_6", "kier_5",
             "kier_4", "kier_3", "kier_2",
             "karo_7", "karo_6", "karo_5", "karo_8", "karo_9", "karo_10", "karo_6", "karo_5", "karo_4", "karo_3", "karo_2",
             "karo_walet", "karo_dama", "karo_krol", "karo_as",
             "pik_as", "pik_krol", "pik_dama", "pik_walet", "pik_10", "pik_9", "pik_8", "pik_7", "pik_6", "pik_5",
             "pik_4", "pik_3", "pik_2"]

SUIT_NAME = ["trefl", "karo", "kier", "pik"]
RANK_NAME = ["as", "krol", "dama", "walet", "10", "9", "8", "7", "6", "5", "4", "3", "2"]

class Rank:
    def __init__(self, image):
        self.image = image
class Suit:
    def __init__(self, image):
        self.image = image
class Card:
    def __init__(self, image):
        self.image = image

def imread_cards(cards_name):
    folder_path = "./patterns/"
    extension = ".jpg"
    cards = {}
    for card_name in cards_name:
        card_image = cv2.imread(folder_path + card_name + extension)
        card = Card(card_image)
        cards[card_name] = card
    return cards

def imread_suit(suits_name):
    folder_path = "./suit/"
    extension = ".jpg"
    suits = {}
    for suit_name in suits_name:
        suit_image = cv2.imread(folder_path + suit_name + extension)
        suit_image = cv2.cvtColor(suit_image, cv2.COLOR_BGR2GRAY)
        suit = Suit(suit_image)
        suits[suit_name] = suit
    return suits

def imread_rank(ranks_name):
    folder_path = "./rank/"
    extension = ".jpg"
    ranks = {}
    for rank_name in ranks_name:
        rank_image = cv2.imread(folder_path + rank_name + extension)
        rank_image = cv2.cvtColor(rank_image, cv2.COLOR_BGR2GRAY)
        rank = Rank(rank_image)
        ranks[rank_name] = rank
    return ranks


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
    print("Ile wykrylo prostokatow w klatce: ", count)
    # Extract the region of interest (ROI) for each card
    cards = []
    for card_contour in card_contours:
        x, y, w, h = cv2.boundingRect(card_contour)
        if w < MIN_CARD_WIDTH or w > MAX_CARD_WIDTH or h < MIN_CARD_HEIGHT or h > MAX_CARD_HEIGHT:
            continue

        print("Dane o wykrytej karcie x: ", x, "y:", y, "w:", w, "h: ", h)
        print("\n")

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

    img_w, img_h = np.shape(card)[:2]

    bkg_level = gray[int(img_w/100)][int(img_h/2)]
    thresh_level = bkg_level - 30
    print("Thresh_level fragmentu karty wynosi: ", thresh_level)

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)

    return thresh

#### Funkcja zwraca kartę wzorcową, która wg niej pasuje do dostarczonej karty z ramki wideo ###
def match_card(card, suits, ranks, cards_patern):
    similarity_suit = 99999  # im mniejsza wartosc tym karty podobne
    similarity_rank = 99999

    match_suit_name = None
    match_rank_name = None
    match_card_name = None

    card_rank, card_suit = get_card_signature(card)

    #porównuje i szukam dopasowania w kolorze karty
    for patern_suit_name in SUIT_NAME:
        patern = cv2.resize(suits[patern_suit_name].image, (card_suit.shape[1],card_suit.shape[0]))
        diff_between_signatures = calc_similarity_between_signatures(card_suit, patern)
        if similarity_suit > diff_between_signatures:
            similarity_suit = diff_between_signatures
            match_suit_name = patern_suit_name

    # porównuje i szukam dopasowania w figurze karty
    for patern_rank_name in RANK_NAME:
        patern = cv2.resize(ranks[patern_rank_name].image, (card_rank.shape[1],card_rank.shape[0]) )
        diff_between_signatures = calc_similarity_between_signatures(card_rank, patern)
        if similarity_rank > diff_between_signatures:
            similarity_rank = diff_between_signatures
            match_rank_name = patern_rank_name

    patern_card_name =  match_suit_name + "_" + match_rank_name
    return cards_patern[patern_card_name].image


def calc_similarity_between_signatures(image_signature, pattern_signature):
    diff_img = cv2.absdiff(image_signature, pattern_signature)
    rank_diff = int(np.sum(diff_img) / 255)
    return rank_diff


def get_card_signature(card):
    part = card[0:CARD_SIGNATURE_HEIGHT, 0:CARD_SIGNATURE_WIDTH]
    part = preprocess_card(part)

    # Sample known white pixel intensity to determine good threshold level
    white_level = part[15, int(CARD_SIGNATURE_WIDTH / 2)]
    thresh_level = white_level - 30
    print(thresh_level)
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(part, thresh_level, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("card signature", query_thresh)
    cv2.waitKey(500)

    # Split in to top and bottom half (top shows rank, bottom shows suit)
    suit = query_thresh[85 : CARD_SIGNATURE_HEIGHT , 0:CARD_SIGNATURE_WIDTH]
    rank = query_thresh[0 : 85 , 0:CARD_SIGNATURE_WIDTH]

    # Find rank contour and bounding rectangle, isolate and find largest contour
    rank_contour, _ = cv2.findContours(rank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rank_contour = sorted(rank_contour, key=cv2.contourArea, reverse=True)
    if len(rank_contour) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(rank_contour[0])
        rank = rank[y1:y1 + h1, x1:x1 + w1]

    # Find suit contour and bounding rectangle, isolate and find largest contour
    suit_contour, _ = cv2.findContours(suit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    suit_contour = sorted(suit_contour, key=cv2.contourArea, reverse=True)
    if len(suit_contour) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(suit_contour[0])
        suit = suit[y1:y1 + h1, x1:x1 + w1]

    return rank, suit


def show_card_from_frame(frame, suits, ranks, cards_patern):

    height, width, channels = frame.shape

    left_border = int(0 + width/4)
    right_border = int(width - width/4)
    new_frame = frame[:, left_border : right_border]

    image1 = preprocess_image(new_frame)

    cards, card_contours = find_cards(image1, new_frame)

    for card, contour in zip(cards, card_contours):
        cv2.imshow("Card", card)
        #wyświetlanie karty wzorcowej
        card_patern = match_card(card, suits, ranks, cards_patern)
        cv2.imshow("Wzorzec", card_patern)
        cv2.waitKey(100)


############ wgrywanie wzorców wykorzystywane do póżniejszej analizy #########

suits = imread_suit(SUIT_NAME)
ranks = imread_rank(RANK_NAME)
cards_pattern = imread_cards(CARDS_NAME)

#################   MAIN - wersja statyczna z pojedynczym zdjęciem ###########################################

# image_filename = "./images/trefl_10.jpg"
# image_rank = "2"
# image_suit = "trefl"
# extension = ".jpg"
#
# image = cv2.imread(image_filename)
# image_after_preprocess = preprocess_image(image)
#
# resized_image = cv2.resize(image_after_preprocess, (960, 540))
# cv2.imshow("Image", resized_image)
# cv2.waitKey(0)
#
# cards, card_contours = find_cards(image_after_preprocess, image)
#
# for card, contour in zip(cards, card_contours):
#     cv2.imshow("Card", card)
#     cv2.waitKey(0)
#
#     rank, suit = get_card_signature(card)
#     cv2.imshow("Card Signature", rank)
#     cv2.waitKey(0)
#     cv2.imshow("Card Signature", suit)
#     cv2.waitKey(0)
#
#     match_card(card, suits, ranks)
#     # cv2.waitKey(0)
#
#     # cv2.imwrite("./rank/"+image_rank+extension, rank)
#     # cv2.imwrite("./suit/"+image_suit+extension, suit)
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

frame_indx = 0 #służy do określenia co którą klatkę ma pobierać obraz do analizy

while True:
    ret, frame = cap.read()
    if not ret:
        print("Koniec pliku wideo.")
        break

    if frame_indx == 10:
        show_card_from_frame(frame, suits, ranks, cards_pattern)
        frame_indx = 0

    resized_frame = cv2.resize(frame, (frame_width, frame_height))
    left_border = int(frame_width/4)
    right_border = int(frame_width - frame_width / 4)
    cv2.line(resized_frame, (left_border, 0), (left_border, resized_frame.shape[0]), (0, 255, 0), 2)
    cv2.line(resized_frame, (right_border, 0), (right_border, resized_frame.shape[0]), (0, 255, 0), 2)

    cv2.imshow('Frame', resized_frame)
    frame_indx += 1

    if cv2.waitKey(50) & 0xFF == ord('q'):  # Czekaj 50 ms między klatkami
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from skimage import feature
from skimage.metrics import structural_similarity as ssim

def preprocess_card(card):
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    img_w, img_h = np.shape(card)[:2]

    bkg_level = gray[int(img_w/100)][int(img_h/2)]
    thresh_level = bkg_level - 30
    print("Thresh_level fragmentu karty wynosi: ", thresh_level)

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)

    return thresh

class CardDetector:
    def __init__(self):
        # ORB (Oriented FAST and Rotated BRIEF) feature detector and the brute-force matcher
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def preprocess_image(self, image):
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def compare_color_histogram(self, image, template):
        image_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        #4th argument maybe will need to be adjusted
        template_hist = cv2.calcHist([template], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        score = cv2.compareHist(image_hist, template_hist, cv2.HISTCMP_CORREL)
        return score

    def compare_shapes(self, image, template):
        image_canny = self.preprocess_image(image)
        template_canny = self.preprocess_image(template)
        contours_image, _ = cv2.findContours(image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_template, _ = cv2.findContours(template_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #return amount of contours
        return len(contours_image), len(contours_template)

    def compare_texture(self, image, template):
        lbp_image = feature.local_binary_pattern(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), P=24, R=3, method="uniform")
        lbp_template = feature.local_binary_pattern(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), P=24, R=3,
                                                    method="uniform")
        lbp_hist_image, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 27), range=(0, 26))
        lbp_hist_template, _ = np.histogram(lbp_template.ravel(), bins=np.arange(0, 27), range=(0, 26))
        lbp_hist_image = lbp_hist_image.astype("float")
        lbp_hist_image /= (lbp_hist_image.sum() + 1e-6)
        lbp_hist_template = lbp_hist_template.astype("float")
        lbp_hist_template /= (lbp_hist_template.sum() + 1e-6)
        return np.sum((lbp_hist_image - lbp_hist_template) ** 2)

    def compare_features(self, image, template):
        #ORB
        keypoints1, descriptors1 = self.orb.detectAndCompute(image, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(template, None)
        matches = self.matcher.match(descriptors1, descriptors2)
        return len(matches)

    def resize_image_to_match(self, image, reference_shape):
        #maybe we won't need that function when we will use camera instead of classic photos
        resized_image = cv2.resize(image, (reference_shape[1], reference_shape[0]))
        return resized_image

    def compare_structure(self, image, template):
        #structural similarity (brightness, structure, contrast
        if image.shape != template.shape:
            image = self.resize_image_to_match(image, template.shape)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(gray_image, gray_template, full=True)
        return score

    def detect_playing_card(self, image, template, similarity_threshold):
        # image = cv2.imread(image_path)
        # template = cv2.imread(template_path)

        processed_image = image #self.preprocess_image(image)
        processed_template = template #self.preprocess_image(template)

        color_similarity = self.compare_color_histogram(image, template)
        shape_similarity = self.compare_shapes(processed_image, processed_template)
        texture_similarity = self.compare_texture(image, template)
        feature_similarity = self.compare_features(image, template)
        structure_similarity = self.compare_structure(image, template)

        processed_image = preprocess_card(image)
        processed_template = preprocess_card(template)

        diff_img = cv2.absdiff(processed_image, processed_template)
        rank_diff = int(np.sum(diff_img) / 255)
        print("RANK DIFF: ", rank_diff)

        print(color_similarity)
        print(texture_similarity)
        print(feature_similarity)
        print(structure_similarity)

        if rank_diff < 10000:
            return True, None
        else:
            return False, None

        # if all([color_similarity > similarity_threshold,
        #         texture_similarity < (1 - similarity_threshold),
        #         # For texture lower value means bigger probability
        #         feature_similarity > 80
        #         ]):
        #     return True, image
        # else:
        #     return False, None

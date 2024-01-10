import cv2

class CameraCapture:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError("Nie można nawiązać połączenia z kamerą.")

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            raise ValueError("Kamera nie jest otwarta.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Nie można odczytać ramki z kamery.")
        return frame

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.release_camera()

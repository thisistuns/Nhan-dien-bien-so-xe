import tkinter as tk
from tkinter import Label, Button, filedialog
from PIL import Image, ImageTk
import cv2
import recognition
import os
import time

# --- Các tham số nhận diện ---
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
Min_char_area = 0.015
Max_char_area = 0.06
Min_char = 0.01
Max_char = 0.09
Min_ratio_char = 0.25
Max_ratio_char = 0.7
max_size_plate = 18000
min_size_plate = 5000
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống nhận diện biển số xe")
        self.video_label = Label(root)
        self.video_label.pack()
        self.plate_label = Label(root)
        self.plate_label.pack()
        self.result_label = Label(root, font=("Arial", 20), fg="blue")
        self.result_label.pack()
        self.btn_rec_video = Button(root, text="Nhận diện video", command=self.recognize_video)
        self.btn_rec_video.pack(side=tk.LEFT, padx=10, pady=10)
        self.btn_rec_image = Button(root, text="Nhận diện ảnh", command=self.recognize_image)
        self.btn_rec_image.pack(side=tk.LEFT, padx=10, pady=10)
        self.btn_rec_camera = Button(root, text="Nhận diện camera", command=self.start_camera)
        self.btn_rec_camera.pack(side=tk.LEFT, padx=10, pady=10)
        self.btn_capture = Button(root, text="Chụp", command=self.capture_frame, state=tk.DISABLED)
        self.btn_capture.pack(side=tk.LEFT, padx=10, pady=10)
        self.cap = None
        self.running = False
        self.current_frame = None
        self.kNearest = recognition.load_knn_model()

    def recognize_video(self):
        video_path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if video_path:
            # Nhận diện và lấy danh sách biển số
            plates = recognition.recognize_video(video_path, self.kNearest)
            if plates:
                self.result_label.config(text="\n".join(plates))
                # Hiển thị ảnh biển số đầu tiên nếu có
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, img = cap.read()
                    if not ret or img is None:
                        break
                    plate_img, plate_text = recognition.detect_plate(img, self.kNearest)
                    if plate_img is not None and plate_text == plates[0]:
                        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                        plate_pil = Image.fromarray(plate_img)
                        plate_pil = plate_pil.resize((320, 100))
                        plate_imgtk = ImageTk.PhotoImage(image=plate_pil)
                        self.plate_label.imgtk = plate_imgtk
                        self.plate_label.configure(image=plate_imgtk)
                        break
                cap.release()
            else:
                self.result_label.config(text="Không phát hiện biển số nào")
                self.plate_label.configure(image=None)

    def recognize_image(self):
        image_path = filedialog.askopenfilename(
            title="Chọn file ảnh",
            filetypes=[("Image files", "*.jpg *.png *.jpeg"), ("All files", "*.*")]
        )
        if image_path:
            img = cv2.imread(image_path)
            if img is not None:
                plate_img, plate_text = recognition.detect_plate(img, self.kNearest)
                if plate_img is not None:
                    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                    plate_pil = Image.fromarray(plate_img)
                    plate_pil = plate_pil.resize((320, 100))
                    plate_imgtk = ImageTk.PhotoImage(image=plate_pil)
                    self.plate_label.imgtk = plate_imgtk
                    self.plate_label.configure(image=plate_imgtk)
                else:
                    self.plate_label.configure(image=None)
                self.result_label.config(text=plate_text)
            else:
                self.result_label.config(text="Không đọc được ảnh")
                self.plate_label.configure(image=None)

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.btn_capture.config(state=tk.NORMAL)
            self.update_camera_frame()

    def update_camera_frame(self):
        if not self.running or not self.cap:
            return
        ret, img = self.cap.read()
        if not ret or img is None:
            self.stop_video()
            return
        self.current_frame = img.copy()
        img_display = cv2.resize(img, (640, 360))
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.plate_label.configure(image=None)
        self.result_label.config(text="")
        self.root.after(30, self.update_camera_frame)

    def capture_frame(self):
        if self.current_frame is not None:
            # Lưu ảnh gốc vào data/image
            if not os.path.exists('data/image'):
                os.makedirs('data/image')
            filename = f"data/image/capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.current_frame)
            plate_img, plate_text = recognition.detect_plate(self.current_frame, self.kNearest)
            if plate_img is not None:
                plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                plate_pil = Image.fromarray(plate_img)
                plate_pil = plate_pil.resize((320, 100))
                plate_imgtk = ImageTk.PhotoImage(image=plate_pil)
                self.plate_label.imgtk = plate_imgtk
                self.plate_label.configure(image=plate_imgtk)
            else:
                self.plate_label.configure(image=None)
            self.result_label.config(text=plate_text)
            # Dừng camera (thay cho self.stop_video())
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.btn_capture.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
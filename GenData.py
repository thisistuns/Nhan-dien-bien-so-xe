# GenData.py
# Tạo dữ liệu huấn luyện cho nhận diện ký tự biển số xe
# Code đã được viết lại rõ ràng, dễ hiểu, có chú thích tiếng Việt

import numpy as np
import cv2
import sys
import os

# Kích thước ảnh ký tự sau khi resize
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
# Diện tích contour tối thiểu để được coi là ký tự
MIN_CONTOUR_AREA = 40

# Danh sách mã ASCII các ký tự hợp lệ (0-9, A-Z)
VALID_CHARS = [ord(c) for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ']

def main():
    # Đọc ảnh chứa các ký tự mẫu
    img = cv2.imread("training_chars.png")
    if img is None:
        print("Không tìm thấy file training_chars.png!")
        return

    # Chuyển sang ảnh xám và làm mờ
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Nhị phân hóa ảnh bằng adaptive threshold
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    cv2.imshow("Ảnh nhị phân", img_thresh)

    # Tìm contour các ký tự
    img_thresh_copy = img_thresh.copy()
    contours, _ = cv2.findContours(img_thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mảng lưu dữ liệu ảnh và nhãn
    flattened_images = []
    classifications = []

    # Duyệt từng contour để lấy ký tự
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            # Vẽ khung đỏ quanh ký tự
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Cắt ký tự và resize về kích thước chuẩn
            img_roi = img_thresh[y:y + h, x:x + w]
            img_resized = cv2.resize(img_roi, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            # Hiển thị ký tự để người dùng nhập nhãn
            cv2.imshow("Ký tự cắt ra", img_roi)
            cv2.imshow("Ký tự đã resize", img_resized)
            cv2.imshow("Ảnh gốc", img)
            key = cv2.waitKey(0)
            if key == 27:  # Nhấn ESC để thoát
                print("Đã thoát!")
                sys.exit()
            elif key in VALID_CHARS:
                classifications.append(key)
                flattened_images.append(img_resized.reshape(-1))
            # Nếu nhấn phím không hợp lệ thì bỏ qua

    # Chuyển sang numpy array
    classifications = np.array(classifications, np.float32).reshape(-1, 1)
    flattened_images = np.array(flattened_images, np.float32)

    # Lưu ra file
    np.savetxt("classifications.txt", classifications)
    np.savetxt("flattened_images.txt", flattened_images)
    print("\n\nĐã lưu dữ liệu huấn luyện vào classifications.txt và flattened_images.txt!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

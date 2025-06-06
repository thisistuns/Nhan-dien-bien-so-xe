# Preprocess.py

import cv2
import numpy as np

# ======================================================================
# CÁC THÔNG SỐ TIỀN XỬ LÝ
# ======================================================================
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)      # Kích thước bộ lọc Gaussian Blur
ADAPTIVE_THRESH_BLOCK_SIZE = 19           # Kích thước vùng tính ngưỡng cục bộ
ADAPTIVE_THRESH_WEIGHT = 9                # Tham số điều chỉnh ngưỡng

# ======================================================================
# HÀM CHÍNH: Tiền xử lý ảnh
# ======================================================================
def preprocess(imgOriginal):
    """
    Tiền xử lý ảnh đầu vào để tách biển số xe gồm:
        - Chuyển sang ảnh xám theo giá trị sáng (HSV)
        - Tăng tương phản cục bộ bằng Top-Hat và Black-Hat
        - Làm mịn ảnh
        - Nhị phân hóa ảnh bằng adaptive threshold
    Trả về:
        - imgGrayscale: ảnh xám
        - imgThresh: ảnh nhị phân (biển số nổi bật)
    """
    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrast = maximizeContrast(imgGrayscale)

    imgBlurred = cv2.GaussianBlur(imgMaxContrast, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv2.adaptiveThreshold(
        imgBlurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_THRESH_BLOCK_SIZE,
        C=ADAPTIVE_THRESH_WEIGHT
    )

    return imgGrayscale, imgThresh

# ======================================================================
# TÁCH KÊNH GIÁ TRỊ (VALUE) TỪ HỆ MÀU HSV
# ======================================================================
def extractValue(imgOriginal):
    """
    Trích xuất kênh Value từ ảnh HSV - biểu thị độ sáng.
    Đây là kênh phù hợp để xử lý biển số vì nhạy với ánh sáng hơn RGB.
    """
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    _, _, imgValue = cv2.split(imgHSV)
    return imgValue

# ======================================================================
# TĂNG CƯỜNG TƯƠNG PHẢN CỤC BỘ
# ======================================================================
def maximizeContrast(imgGrayscale):
    """
    Tăng tương phản ảnh xám bằng cách kết hợp Top-Hat và Black-Hat morphology.
    Mục tiêu là làm nổi bật các đặc trưng như chữ và đường viền trên biển số.
    """
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(
        imgGrayscale,
        cv2.MORPH_TOPHAT,
        structuringElement,
        iterations=10
    )

    imgBlackHat = cv2.morphologyEx(
        imgGrayscale,
        cv2.MORPH_BLACKHAT,
        structuringElement,
        iterations=10
    )

    imgContrastEnhanced = cv2.add(imgGrayscale, imgTopHat)
    imgContrastEnhanced = cv2.subtract(imgContrastEnhanced, imgBlackHat)

    return imgContrastEnhanced

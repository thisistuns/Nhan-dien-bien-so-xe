import cv2
import numpy as np
import math
import Preprocess

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
Min_char = 0.01
Max_char = 0.09
Min_ratio_char = 0.25
Max_ratio_char = 0.7

# Hàm load model KNN

def load_knn_model():
    npaClassifications = np.loadtxt("classifications.txt", np.float32)
    npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    kNearest = cv2.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    return kNearest

# Hàm nhận diện biển số

def detect_plate(img, kNearest):
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h
        if (len(approx) == 4) and (0.5 <= ratio <= 7.0):
            screenCnt.append(approx)
    if not screenCnt:
        return None, "Không phát hiện biển số"
    for cnt in screenCnt:
        (x1, y1) = cnt[0, 0]
        (x2, y2) = cnt[1, 0]
        (x3, y3) = cnt[2, 0]
        (x4, y4) = cnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        if ke == 0:
            continue
        angle = math.atan(doi / ke) * (180.0 / math.pi)
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [cnt], 0, 255, -1)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        roi = img[topx:bottomx + 1, topy:bottomy + 1]
        imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2
        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)
        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width
        for ind, cnt2 in enumerate(cont):
            area = cv2.contourArea(cnt2)
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind
        if len(char_x) >= 5:
            char_x = sorted(char_x)
            first_line = ""
            second_line = ""
            for i in char_x:
                (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                imgROI = thre_mor[y:y + h, x:x + w]
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaROIResized = np.float32(npaROIResized)
                _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=3)
                strCurrentChar = str(chr(int(npaResults[0][0])))
                if (y < height / 3):
                    first_line = first_line + strCurrentChar
                else:
                    second_line = second_line + strCurrentChar
            strFinalString = first_line + " - " + second_line
            return roi, strFinalString
    return None, "Không phát hiện biển số"

def recognize_video(video_path, kNearest):
    cap = cv2.VideoCapture(video_path)
    tongframe = 0
    biensotimthay = 0
    plates = []
    while cap.isOpened():
        ret, img = cap.read()
        if not ret or img is None:
            break
        tongframe += 1
        roi, plate_text = detect_plate(img, kNearest)
        if roi is not None and plate_text != "Không phát hiện biển số":
            plates.append(plate_text)
            biensotimthay += 1
            # Hiển thị biển số lên ảnh
            cv2.imshow("License Plate", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        # Hiển thị frame
        imgcopy = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow('Video', imgcopy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Tổng số frame: {tongframe}, Biển số tìm thấy: {biensotimthay}")
    return plates

def recognize_image(image_path, kNearest):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không đọc được ảnh: {image_path}")
        return []
    img = cv2.resize(img, dsize=(1920, 1080))
    roi, plate_text = detect_plate(img, kNearest)
    plates = []
    if roi is not None and plate_text != "Không phát hiện biển số":
        plates.append(plate_text)
        cv2.imshow("License Plate", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow('License plate', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return plates

def recognize_camera(kNearest):
    cap = cv2.VideoCapture(0)
    tongframe = 0
    biensotimthay = 0
    plates = []
    while cap.isOpened():
        ret, img = cap.read()
        if not ret or img is None:
            break
        tongframe += 1
        roi, plate_text = detect_plate(img, kNearest)
        if roi is not None and plate_text != "Không phát hiện biển số":
            plates.append(plate_text)
            biensotimthay += 1
            # Hiển thị biển số lên ảnh
            cv2.imshow("License Plate", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        # Hiển thị frame với kết quả nhận diện
        imgcopy = cv2.resize(img, None, fx=0.5, fy=0.5)
        if plate_text and plate_text != "Không phát hiện biển số":
            cv2.putText(imgcopy, plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('Camera', imgcopy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Tổng số frame: {tongframe}, Biển số tìm thấy: {biensotimthay}")
    return plates 
# Hệ thống nhận diện biển số xe

## Giới thiệu
Đây là dự án nhận diện biển số xe sử dụng Python, OpenCV và giao diện Tkinter. Hệ thống cho phép nhận diện biển số từ ảnh, video hoặc trực tiếp từ camera, đồng thời hỗ trợ tạo dữ liệu huấn luyện cho mô hình nhận diện ký tự biển số.

## Cấu trúc thư mục
```
├── gui_app.py                # Ứng dụng giao diện người dùng (GUI)
├── recognition.py            # Xử lý nhận diện biển số và ký tự
├── Preprocess.py             # Tiền xử lý ảnh biển số
├── GenData.py                # Tạo dữ liệu huấn luyện ký tự
├── classifications.txt       # Nhãn ký tự dùng cho KNN
├── flattened_images.txt      # Dữ liệu ảnh ký tự đã làm phẳng
├── training_chars.png        # Ảnh ký tự dùng để tạo dữ liệu huấn luyện
├── requirements.txt          # Danh sách thư viện cần thiết
├── data/
│   └── image/                # Ảnh chụp từ camera sẽ lưu tại đây
├── result/                   # Lưu các ảnh kết quả xử lý, debug
```

## Cài đặt
1. **Cài Python 3.7
2. **Cài các thư viện cần thiết:**
   ```sh
   python3 -m pip install -r requirements.txt
   ```
   Nếu chỉ cần chạy GUI, bạn có thể cài nhanh các thư viện chính:
   ```sh
   python3 -m pip install opencv-python numpy Pillow
   ```
3. **Kiểm tra các file dữ liệu:**
   - Đảm bảo có `classifications.txt` và `flattened_images.txt` trong thư mục gốc.
   - Nếu chưa có, xem mục "Tạo dữ liệu huấn luyện" bên dưới.

## Hướng dẫn sử dụng giao diện (GUI)
Chạy ứng dụng bằng lệnh:
```sh
python3 gui_app.py
```
Các chức năng chính:
- **Nhận diện ảnh:** Chọn ảnh biển số, hệ thống sẽ hiển thị kết quả nhận diện.
- **Nhận diện video:** Chọn file video, hệ thống sẽ quét và nhận diện biển số trong các khung hình.
- **Nhận diện camera:** Mở webcam, nhấn "Chụp" để nhận diện biển số từ khung hình hiện tại.
- **Kết quả:** Biển số nhận diện được sẽ hiển thị trên giao diện.
- Ảnh chụp từ camera sẽ lưu vào thư mục `data/image/`.

## Mô tả các file chính
- **gui_app.py:** Giao diện người dùng, cho phép thao tác nhận diện biển số từ nhiều nguồn.
- **recognition.py:** Chứa các hàm nhận diện biển số, nhận diện ký tự, và xử lý video/camera.
- **Preprocess.py:** Tiền xử lý ảnh (làm mịn, tăng tương phản, nhị phân hóa).
- **GenData.py:** Script tạo dữ liệu huấn luyện ký tự từ ảnh `training_chars.png`.
- **classifications.txt, flattened_images.txt:** Dữ liệu huấn luyện cho mô hình KNN nhận diện ký tự.
- **training_chars.png:** Ảnh chứa các ký tự mẫu để tạo dữ liệu huấn luyện.
- **data/image/:** Lưu ảnh chụp từ camera.
- **result/:** Lưu các ảnh kết quả, debug trong quá trình phát triển.

## Tạo dữ liệu huấn luyện ký tự
Nếu chưa có `classifications.txt` và `flattened_images.txt`, bạn cần tạo bằng cách:
1. Chuẩn bị ảnh `training_chars.png` chứa các ký tự mẫu (0-9, A-Z).
2. Chạy script tạo dữ liệu:
   ```sh
   python3 GenData.py
   ```
3. Làm theo hướng dẫn trên màn hình để gán nhãn cho từng ký tự.
4. Sau khi hoàn thành, hai file dữ liệu sẽ được tạo trong thư mục gốc.

## Lưu ý khi chạy trên macOS
- Luôn dùng `python3` thay vì `python`.
- Nếu thiếu thư viện, cài bằng `python3 -m pip install ...`.
- Nếu gặp lỗi về giao diện (Tkinter), đảm bảo Python cài qua Homebrew đã có Tkinter.
- Nếu dùng camera, cấp quyền truy cập camera cho Terminal.

## Liên hệ
Tác giả: Đào Anh Tuấn, Đỗ Đắc Nhật, Dũng

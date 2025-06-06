# Phân Loại Bệnh Táo Bằng MobileNetV2

Dự án này nhằm mục đích phân loại các bệnh phổ biến trên lá táo dựa trên hình ảnh, sử dụng mô hình MobileNetV2 đã được huấn luyện trước (pre-trained).

## Dataset

Sử dụng bộ dữ liệu **New Plant Diseases Dataset**.

Các lớp (classes) được sử dụng để phân loại:

1.  `Apple___Apple_scab`
2.  `Apple___Black_rot`
3.  `Apple___Cedar_apple_rust`
4.  `Apple___healthy` (Lá táo khỏe mạnh)

## Giải thích Mã Nguồn

### 1. Chuẩn bị Dữ liệu (ImageDataGenerator)

Bước này chuẩn bị dữ liệu hình ảnh đầu vào cho mô hình.

* **`IMG_SIZE = 224`**: Tất cả hình ảnh đầu vào sẽ được điều chỉnh về kích thước 224x224 pixel.
* **`BATCH_SIZE = 32`**: Dữ liệu sẽ được nạp vào mô hình theo từng lô 32 ảnh.
* **`ImageDataGenerator`**:
    * `rescale=1./255`: Chuẩn hóa giá trị của các pixel trong ảnh về khoảng từ 0 đến 1.
    * `validation_split=0.2`: Phân chia 20% dữ liệu từ thư mục gốc làm tập kiểm định (validation), 80% còn lại được sử dụng cho tập huấn luyện (training).
* **`train_gen`**: Đối tượng generator để tải dữ liệu cho tập huấn luyện.
* **`val_gen`**: Đối tượng generator để tải dữ liệu cho tập kiểm định.

Cả `train_gen` và `val_gen` đều đọc ảnh từ thư mục được chỉ định trong biến `base_dir`.

### 2. Xây dựng Mô hình (Sử dụng MobileNetV2)

* **`MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))`**:
    * Tải kiến trúc `MobileNetV2` với trọng số đã được huấn luyện trước trên bộ dữ liệu `ImageNet`.
    * `include_top=False`: Loại bỏ lớp phân loại cuối cùng (fully-connected layer) của mô hình MobileNetV2 gốc. Điều này cho phép tùy chỉnh phần đầu ra của mô hình cho bài toán cụ thể.
    * `input_shape=(IMG_SIZE,IMG_SIZE,3)`: Định nghĩa kích thước ảnh đầu vào là 224x224 pixel với 3 kênh màu (RGB).
* **`base_model.trainable=False`**: Đóng băng các trọng số của mô hình `MobileNetV2` gốc. Trong quá trình huấn luyện ban đầu, chỉ các trọng số của những lớp được thêm vào sau này sẽ được cập nhật.

### 3. Thêm các Lớp Đầu ra Tùy chỉnh

Các lớp sau được thêm vào phía trên `base_model` để phù hợp với bài toán phân loại bệnh táo:

* **`x = base_model.output`**: Lấy đầu ra từ `base_model`.
* **`x = GlobalAveragePooling2D()(x)`**: Áp dụng lớp Global Average Pooling để giảm số lượng tham số và kích thước đặc trưng.
* **`x = Dense(128, activation='relu')(x)`**: Thêm một lớp fully connected (Dense) với 128 đơn vị và hàm kích hoạt ReLU.
* **`pred = Dense(4, activation='softmax')(x)`**: Lớp đầu ra cuối cùng với 4 đơn vị (tương ứng với 4 lớp bệnh) và hàm kích hoạt `softmax` để đưa ra xác suất cho mỗi lớp.

### 4. Hoàn thiện và Biên dịch (Compile) Mô hình

* **`model = Model(inputs=base_model.input, outputs=pred)`**: Tạo mô hình cuối cùng bằng cách kết nối đầu vào của `base_model` với lớp đầu ra `pred`.
* **`model.compile(...)`**: Cấu hình quá trình học cho mô hình:
    * `optimizer=Adam()`: Sử dụng trình tối ưu hóa Adam.
    * `loss='categorical_crossentropy'`: Sử dụng hàm mất mát categorical crossentropy, phù hợp cho bài toán phân loại đa lớp với đầu ra dạng one-hot encoding.
    * `metrics=['accuracy']`: Theo dõi độ chính xác (accuracy) trong quá trình huấn luyện và đánh giá.

## Cách thực thi

1.  **Cài đặt các thư viện cần thiết:**
    Mở terminal hoặc command prompt, di chuyển đến thư mục dự án và chạy lệnh:
    ```bash
    pip install -r requirements.txt
    ```
    (Lưu ý: `requirements.txt` cần chứa danh sách các thư viện như `tensorflow`, `numpy`, `matplotlib`, v.v.)

2.  **Chạy kịch bản chính:**
    ```bash
    python run.py
    ```
    Hoặc nếu sử dụng phiên bản Python cụ thể:
    ```bash
    py run.py
    ```

3.  **Truy cập ứng dụng (nếu có giao diện web):**
    Mở trình duyệt web và điều hướng đến: `localhost:5000/lession-ann`

## Hiển thị Biểu đồ Huấn luyện/Kiểm định

<img src="https://github.com/HitDrama/Apple-Disease-MobileNetV2-/blob/main/static/train/train-plant-disease.png" alt="Training Chart" width="100%"/>
<img src="https://github.com/HitDrama/Apple-Disease-MobileNetV2-/blob/main/static/train/test.png" alt="Validation/Testing Chart" width="45%"/>


---

👨‍💻 Người phát triển: Đặng Tố Nhân  
🔧 Stack chính: Python · TensorFlow · Flask · Keras · HTML/CSS

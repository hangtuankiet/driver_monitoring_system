# Driver Monitoring System - Model Selection Update

## Những thay đổi mới nhất:

### 1. Lazy Loading của Models

- **Không load models khi khởi động**: Models sẽ chỉ được load khi cần thiết, không phải ngay khi khởi động
- **Load models khi bắt đầu monitoring**: Models chỉ được load khi bắt đầu camera hoặc video monitoring
- **Cấu hình được lưu ngay lập tức**: Khi thay đổi model trong Settings, chỉ cấu hình được cập nhật, model thực tế chỉ được load khi cần

### 2. Lựa chọn đa dạng Model trong GUI:

- **YOLO Models** (Detection):

  - YOLOv10 (`models/detected/yolov10.pt`)
  - YOLOv11 (`models/detected/yolov11.pt`)

- **Classification Backbones** (Drowsiness):
  - VGG16 (`models/classification/vgg16_model.pth`)
  - MobileNet V2 (`models/classification/mobilenet_v2_model.pth`)
  - MobileNet V3 Small (`models/classification/mobilenet_v3_small_model.pth`)
  - EfficientNet B0 (`models/classification/efficientnet_b0_model.pth`)
- Thêm `get_current_backbone()` và `get_available_backbones()`

### 4. Cập nhật `src/gui.py`:

- Thêm section "Model Settings" trong cửa sổ Settings
- Thêm combobox để chọn backbone
- Cập nhật logic save settings để thay đổi model

## Hướng dẫn sử dụng:

### 1. Chuẩn bị model files:

Đặt các file model vào thư mục `models/classification/`:

- `vgg16_model.pth`
- `mobilenet_v2_model.pth`
- `mobilenet_v3_small_model.pth`
- `efficientnet_b0_model.pth`

### 2. Chạy ứng dụng:

```bash
python main.py
```

### 3. Thay đổi model:

1. Mở Settings (⚙ Settings button hoặc File > Settings)
2. Trong section "Model Settings", chọn backbone mong muốn
3. Click "Save Settings"
4. Restart monitoring nếu đang chạy

### 4. Các backbone có sẵn:

- **VGG16**: Model mặc định, độ chính xác cao nhưng chậm
- **MobileNet-V2**: Nhẹ, nhanh, phù hợp cho real-time
- **MobileNet-V3-Small**: Rất nhẹ, rất nhanh, ít tài nguyên
- **EfficientNet-B0**: Cân bằng giữa hiệu suất và tốc độ

## Cấu hình mặc định:

```json
{
  "classification_backbone": "vgg16",
  "classification_model_path": "models/classification/vgg16_model.pth",
  "num_classes": 4
}
```

## Notes:

- Cần dừng monitoring trước khi thay đổi model
- Tất cả model phải có cùng số classes (4 classes mặc định)
- Model weights phải tương thích với architecture được định nghĩa trong `get_model()`

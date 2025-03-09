# import h5py
#
# vgg16_model_path = '../model_yawn_new.h5'
#
# # Mở tệp HDF5
# with h5py.File(vgg16_model_path, 'r') as f:
#     print(f.keys())  # In các khóa cấp cao
#     model_config = f.attrs.get('model_config')
#     if model_config:
#         print(model_config)  # In trực tiếp cấu hình mô hình
#     else:
#         print("Không tìm thấy 'model_config' trong tệp HDF5.")
#
#

from tensorflow.keras.models import load_model

vgg16_model_path = '../vgg16_yawn.weights.h5'
try:
    vgg16_model = load_model(vgg16_model_path, compile=False)
    print("Đã load mô hình thành công!")
    vgg16_model.summary()
except Exception as e:
    print(f"Lỗi khi load như mô hình: {e}")
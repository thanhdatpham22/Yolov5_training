import torch
from models.yolo import Model

# Tạo mô hình YOLOv5s
model = Model('yolov5s')

# Hiển thị cấu trúc mạng
print(model)

# Truy cập các thông số kỹ thuật
input_size = model.model[-1].yaml['nc']
num_classes = model.model[-1].yaml['nc']
anchors = model.model[-1].yaml['anchors']
strides = model.model[-1].yaml['strides']
scales = model.model[-1].yaml['scales']

print("Input size:", input_size)
print("Number of classes:", num_classes)
print("Anchors:", anchors)
print("Strides:", strides)
print("Scales:", scales)
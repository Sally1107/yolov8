from ultralytics import YOLO

# Load a model
model = YOLO('yolov8-seg.yaml')  # build a new model from YAML
model = YOLO('ultralytics/yolov8m-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('ultralytics/yolov8m-seg.pt').load('yolov8-seg.yaml')  # build from YAML and transfer weights

# Train the model
results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
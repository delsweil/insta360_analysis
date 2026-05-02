from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data='/Users/davidelsweiler/datasets/ball_detector_merged-1/data.yaml',
    epochs=150,
    imgsz=960,
    batch=4,
    device='mps',
    mosaic=1.0,
    scale=0.5,
    hsv_v=0.4,
    hsv_s=0.7,
    flipud=0.0,
    fliplr=0.5,
    translate=0.1,
    project='runs/ball',
    name='yolov8s_elevated_v1'
)

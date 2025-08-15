from ultralytics import YOLO

model = YOLO("model/yolo12n.pt")

model.train(data="data.yaml", imgsz=640, device=0, batch=8, epochs=100, workers=0)



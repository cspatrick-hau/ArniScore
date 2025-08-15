from ultralytics import YOLO

model = YOLO("results/weights/best.pt")  


model.export(format="onnx")

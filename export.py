from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("runs/detect/custom_camera_yolo9/weights/best.pt")

# Export the model to CoreML format
model.export(format="coreml")  # creates 'yolo11n.mlpackage'

# Load the exported CoreML model
coreml_model = YOLO("runs/detect/custom_camera_yolo9/weights/best.mlpackage")

# Run inference
results = coreml_model("https://static2-images.vnncdn.net/files/publish/2023/1/19/camera-iphone-216.png")
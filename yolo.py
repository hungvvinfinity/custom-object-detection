from ultralytics import YOLO

# 1. Load a pretrained model (e.g., YOLOv8n)
model = YOLO('yolo11s.pt')

# 2. Train the model
results = model.train(
    data='data.yaml',     # path to the dataset config
    epochs=5000,            # number of training epochs
    imgsz=640,            # image size
    batch=-1,             # batch size
    name='custom_camera_yolo',  # experiment name
)

# validate the model
results = model.val()

results = model("cam2.jpeg")
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk


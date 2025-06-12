from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Inference
model = YOLO("PCB_YOLO12n_160Epochs_Best.pt")  # Adjust path as needed
results1 = model("01_missing_hole_01.jpg")     # Ensure this image is in your folder

# Show inference result
results1[0].show()

# Training (optional)
results2 = model.train(
    data="pcb.yaml",   # This uses the updated YAML with ./pcb_defects_yolo path
    epochs=60,
    imgsz=640,
    batch=4,
)

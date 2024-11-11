from ultralytics import YOLO
import os
# Load a model

model = YOLO("yolov8n.pt")
#model = YOLO("yolov8s.pt")
#model = YOLO("yolov8m.pt")
#model = YOLO("yolov8l.pt")
#model = YOLO("yolov8x.pt")

# Train the model
#train_results = model.train(
 #   data="coco128.yaml",  # path to dataset YAML
  #  epochs=10,  # number of training epochs
   # imgsz=640,  # training image size
    #device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
#)
# Evaluate model performance on the validation set
#metrics = model.val()

# Perform object detection on an image
#results = model("https://ultralytics.com/images/cat.jpg",show=True, save=True ,conf=0.4)
#results = model("./vini.mp4",show=True, save=False, conf=0.6)
results = model("./cat.jpg",show=True, save=True)
#results = model(source =0,show=True, save=False,conf=0.4)
# Export the model to ONNX format
#path = model.export(format="onnx")  # return path to exported model





from ultralytics import YOLO

# Load the model
model = YOLO(r"F:\P.H.I For Engineering and Technology\Fourth year communications\Graduation Project\TSR_V1\best.pt")

results = model(source=0,show=True, save=False,conf=0.4)

#results = model("./red_traffic.mp4",show=True,save=True)

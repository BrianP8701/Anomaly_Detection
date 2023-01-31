from roboflow import Roboflow
# import Image

rf = Roboflow(api_key="hYI9Q9apb6kNv3sGv8xP")
project = rf.workspace("brian-przezdziecki-zp5de").project("tipdetectionv1")
# model = project.version(1).download("yolov8")
model = project.version(1).model

# infer on a local image
print(model.predict("/Users/brianprzezdziecki/Research/Mechatronics/Anomaly_Detection/TipDatasetv1/image200.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("/Users/brianprzezdziecki/Research/Mechatronics/Anomaly_Detection/TipDatasetv1/image0.jpg", confidence=40, overlap=30).save("prediction2.jpg")


# Image.open('pathToFile').show()



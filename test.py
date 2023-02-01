import numpy as np
import cv2
import json

img = cv2.imread("/Users/brianprzezdziecki/Research/Mechatronics/Anomaly_Detection/TipDatasetv1/image0.jpg")

data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')

data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')

print(data.shape)



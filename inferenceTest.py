import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper


model="/Users/brianprzezdziecki/Downloads/best.onnx"
# path=sys.argv[1]

img = cv2.imread("/Users/brianprzezdziecki/Research/Mechatronics/Anomaly_Detection/TipDatasetv1/image0.jpg")
img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_AREA)
img.resize((1, 3, 640, 640))

data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')

data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(input_name)
print(output_name)

result = session.run([output_name], {input_name: data})
print(np.array(result).shape)

def detect_yolo_onnx(confidenc, image, input_width, input_height, class_list, model, model_path, model_details):
    INPUT_WIDTH = input_width
    INPUT_HEIGHT = input_height
    class_list = class_list
    net = cv2.dnn.readNet(model_path)
    row, col, _ = image.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = image
    detection_start = time.perf_counter()
    blob = cv2.dnn.blobFromImage(result, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detection_finished = time.perf_counter()
    if model == "yolov8":
        preds = preds.transpose((0, 2, 1))
    class_ids = []
    confidences = []
    boxes = []
    rows = preds[0].shape[0]
    image_width, image_height, _ = result.shape
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    list_of_conf = []
    for r in range(rows):
        row = preds[0][r]
        confidence = row[4]
        list_of_conf.append(confidence)
        if confidence >= confidenc:
            if model == "yolov8":
                classes_scores = row[4:]
            else:
                classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 
    result_class_ids = []
    result_confidences = []
    result_boxes = []
    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
    for (classid, confidence, box) in zip(result_class_ids, result_confidences, result_boxes):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(image, box, color, 5)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(image, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
    yolo_onnx_returned = []
    yolo_onnx_returned.append(model_details)
    if len(result_boxes) == 0:
        yolo_onnx_returned.append(f"Detection attempt took {round(detection_finished - detection_start,2)} seconds. No tattos detected. Try setting confindence level to {(round(np.amax(list_of_conf) * 100) -1)} or lower.")
    else:
        yolo_onnx_returned.append(f"Model detected {len(result_boxes)} tattoos in {round(detection_finished - detection_start,2)} seconds with a confidence level of {confidenc * 100} percent.")
    yolo_onnx_returned.append(image)
    models_returned.append(yolo_onnx_returned)
    return models_returned

# To run on pt file
# yolo task=detect mode=predict model="/Users/brianprzezdziecki/Downloads/best.pt" source="/Users/brianprzezdziecki/Research/Mechatronics/Anomaly_Detection/TipDatasetv1/image0.jpg" 

# To export pt file to onnx file
# yolo export model="/Users/brianprzezdziecki/Downloads/best.pt" format=onnx 
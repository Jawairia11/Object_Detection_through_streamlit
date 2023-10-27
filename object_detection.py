import cv2
import torch
import numpy as np
import tempfile
import io

def perform_object_detection(uploaded_image, model_path='./yolov7.pt'):
    # Load YOLOv7 model
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path, force_reload=False)
    model.eval()

    # Function to perform object detection and draw bounding boxes
    def detect_objects(image):
        results = model(image)
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.tolist()
            class_name = names[int(cls)]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            image = cv2.putText(image, f"{class_name}: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']

    # Convert BytesIO to OpenCV image
    uploaded_image = io.BytesIO(uploaded_image.getvalue())
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

    detected_image = detect_objects(image)
    return detected_image
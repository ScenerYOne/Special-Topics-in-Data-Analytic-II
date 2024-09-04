import cv2
from ultralytics import YOLO
import os

# Load a pre-trained YOLOv8 model
model = YOLO('D:/Python/Special Topics in Data Analytic II/yolo weight/Yolyo/wight/Yolyotrain3/weights/best.pt')  # Load the smallest YOLOv8 model

# Function to perform object detection
def detect_objects(image_path, annotation_bb):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Perform inference
    results = model(img)

    # Process results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])

            # Draw detected bounding box in green
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw annotated bounding box in blue
    a_x1, a_y1, a_x2, a_y2 = map(int, annotation_bb)
    cv2.rectangle(img, (a_x1, a_y1), (a_x2, a_y2), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Result', img)
    cv2.waitKey(1000)  # Display the image for 1 second

    # Optionally, save the result
    # cv2.imwrite('result.jpg', img)
    print("Detection complete. Bounding boxes are displayed.")
    BoxA = [x1, y1, x2, y2]
    BoxB = [a_x1, a_y1, a_x2, a_y2]
    return BoxA, BoxB

def YOLO2BB(yolo_bb, h_i, w_i):
    # yolo format: cen_x, cen_y, width, height
    # bb: l, t, r, b
    width_h = yolo_bb[2] / 2
    height_h = yolo_bb[3] / 2
    l = yolo_bb[0] - width_h
    t = yolo_bb[1] - height_h
    r = yolo_bb[0] + width_h
    b = yolo_bb[1] + height_h

    return [l * w_i, t * h_i, r * w_i, b * h_i]

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

# Example usage
images_dir = 'Wood cross section/Wood cross section/Images'
annotations_dir = 'Wood cross section/Wood cross section/Annotations'

IoU_all = 0.0
num_images = 0
for image in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image)
    annotation_path = os.path.join(annotations_dir, os.path.splitext(image)[0] + '.txt')
    if image.startswith('.'):
        continue
    with open(annotation_path, 'r') as file:
        data = file.read().strip()

    data = data.split(' ')
    data = [float(i) for i in data]
    data = [data[1], data[2], data[3], data[4]]  # Extracting the YOLO format coordinates
    h, w, _ = cv2.imread(image_path).shape  # Get image dimensions
    annotation_bb = YOLO2BB(data, h, w)  # Convert YOLO format to bounding box
    if image.endswith(('.jpg', '.png', '.jpeg')):
        BB = detect_objects(image_path, annotation_bb)
        BBa, BBb = BB
        iou = bb_intersection_over_union(BBa, BBb)
        print(f"IoU for {image}: {iou:.2f}")
        IoU_all += iou
        num_images += 1

average_iou = IoU_all / num_images
print(f"Average IoU: {average_iou:.2f}")

cv2.destroyAllWindows()

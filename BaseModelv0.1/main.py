from ultralytics import YOLO
import os
import glob
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from model import build_model


def crop_image(box, image):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    x1 = x1 - width * 3 / 2
    x2 = x2 + width * 3 / 2
    y2 = y2 + height * 2
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    return cropped_image, x1, y1, x2, y2


def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return test_transform


def inference(model, testloader, DEVICE):
    model.eval()
    counter = 0
    with torch.no_grad():
        counter += 1
        image = testloader
        image = image.to(DEVICE)
        outputs = model(image)

    predictions = F.softmax(outputs, dim=1).cpu().numpy()
    output_class = np.argmax(predictions)
    # result = annotate_image(image, output_class)
    return output_class


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224
input_path = 'inputs'
all_image_paths = glob.glob(os.path.join(input_path, '*'))
os.makedirs('outputs', exist_ok=True)
yolo = YOLO('model/yolov8n_100e.onnx')
transform = get_test_transform(IMAGE_RESIZE)
checkpoint = torch.load('model/best_model.pth', map_location=DEVICE)
model = build_model(fine_tune=False, num_classes=10).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

for i, image_path in enumerate(all_image_paths):
    orig_image = cv2.imread(image_path)

    with torch.no_grad():
        results = yolo(orig_image, imgsz=320)

    for r in results:
        if len(r.boxes.xyxy) > 0:
            for box in r.boxes.xyxy:
                cropped_image, x1, y1, x2, y2 = crop_image(box, orig_image)
                image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                image = transform(image)
                image = torch.unsqueeze(image, 0)
                result = inference(model, image, DEVICE)
                if result == 1 or result == 2 or result == 3 or result == 4 or result == 9:
                    cv2.rectangle(orig_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(orig_image,
                                "the calling or texting person",
                                (int(x1), int(y1)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2)
    image_name = image_path.split(os.path.sep)[-1]
    cv2.imwrite('outputs/' + image_name, orig_image)
    # cv2.imshow("Output", orig_image)
    # cv2.waitKey(0)

# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import face_recognition

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def real_time_detection(image, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("is Real Face. Score: {:.2f}.".format(value))

        result_text = "Real_Face Score: {:.2f}".format(value)
        color = (0, 255, 0)
    else:
        print("is Fake Face. Score: {:.2f}.".format(value))
        result_text = "Fake_Face Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color, 1)

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        frame = cv2.resize(frame, (640, 480))
        desc = "test"
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument(
            "--device_id",
            type=int,
            default=0,
            help="which gpu id, [0/1/2/3]")
        parser.add_argument(
            "--model_dir",
            type=str,
            default="./resources/anti_spoof_models",
            help="model_lib used to test")
        parser.add_argument(
            "--image_name",
            type=str,
            default=frame,
            help="image used to test")
        args = parser.parse_args()

        
        result = real_time_detection(args.image_name, args.model_dir, args.device_id)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
    cam.release()
    cv2.destroyAllWindows()

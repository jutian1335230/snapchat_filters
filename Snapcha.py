import cv2
import face_recognition
import matplotlib.pyplot as plt
import scipy # another useful library 
import os
import math
import numpy as np
import random

def halo_effect(image, face):
    for (x, y, w, h) in face:
        center = (x+w//2, y+h//2)
        radii = (w//2, h//2)
        num_lines = 36
        line_length = 20 

        for i in range(num_lines):
            if random.randint(0, 5) == 0:
                continue
            angle_rad = math.radians(i * (360 / num_lines))
            start_point = (
                int(random.uniform(1, 1.1) * (center[0] + radii[0] * math.cos(angle_rad))),
                int(random.uniform(1, 1.1) * (center[1] + radii[1] * math.sin(angle_rad)))
            )
            while start_point[0] > 0 and start_point[1] > 0 and start_point[0] < image.shape[1] and start_point[1] < image.shape[0]:
                line_mask = np.zeros_like(image)
                end_point = (
                    int(start_point[0] + line_length * math.cos(angle_rad)),
                    int(start_point[1] + line_length * math.sin(angle_rad))
                )
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.line(line_mask, start_point, end_point, color)
                if random.randint(0, 1) == 0:
                    kernel_size = random.choice([3, 5, 7])
                    line_mask = cv2.GaussianBlur(line_mask, (kernel_size, kernel_size), 0)
                hsv = cv2.cvtColor(line_mask, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.2, 2), 0, 255)
                line_mask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                image += line_mask

                start_point = end_point
                
    return image 
os.chdir(r"C:\Users\tonyt\OneDrive\Documents\GitHub\EE371Q_project\images") 
image = cv2.imread('joey.jpg')
#you can read and turn to gray in one step image_bird = cv2.imread('bird.jpg', cv2.IMREAD_GRAYSCALE) 



gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
halo_image = halo_effect(image, face)
plt.figure(figsize=(20,10))
plt.imshow(halo_image)
plt.axis('off')
plt.show()



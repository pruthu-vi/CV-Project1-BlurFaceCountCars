import cv2
import matplotlib.pyplot as plt
#import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox

# read image
img = cv2.imread("Resources/save_blured_face.jpg")

# find common objects in image
# This will find all the objects in image
bbox, label, conf = cv.detect_common_objects(img)

# drawing bbox around all the object
output_image = draw_bbox(img,bbox,label,conf)
plt.imshow(output_image)
plt.show()

#filename = "Results/bboxed_object.jpg"
#cv2.imwrite(filename, output_image)

# Printing number of cars (simply select key car)
print("Number of cars in the image is "+ str(label.count("car")))
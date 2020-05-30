from keras.models import load_model
import cv2
import os 
import numpy as np
import math

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

model = load_model("model.h5")

smoothed_angle = 0 

def keras_predict(model,image):
    processed = keras_process_image(image)
    angle = model.predict(processed,batch_size=16)
    steering_angle = angle[0][0] * 180/math.pi
    print("Predicted steering angle: " + str(degrees) + " degrees")

    return steering_angle

def keras_process_image(img):
    image_x = 320
    image_y = 160
    img = cv2.resize(img,(image_x,image_y))
    img = np.array(img,dtype=np.float32)
    img = np.reshape(img,(-1,image_x,image_y,3))
    return img 


i=0 

while(cv2.waitKey(10) != ord('q')):
    image = cv2.imread("driving_dataset/" + str(i) + ".jpg")
    image = cv2.resize(image,(320,160))
    degrees = keras_predict(model,image)

    cv2.imshow("frame", image)
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1


cv2.destroyAllWindows()
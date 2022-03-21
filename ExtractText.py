# Import libraries
import PIL.Image
from PIL import Image
import pytesseract
import sys
from pdf2image import convert_from_path
import os
import cv2
from pytesseract import Output


#Create our config

myconfig = r"--psm 12 --oem 3"


#Get the text
#text = pytesseract.image_to_string(PIL.Image.open("Signs.jpg"), config=myconfig)
#print(text)



#Draw rectangles around the recognized characters

img = cv2.imread("Signs.jpg")
height, width, _ = img.shape

###################
#boxes = pytesseract.image_to_boxes(img, config=myconfig)
#for box in boxes.splitlines():
 #   box = box.split(" ")
  #  img = cv2.rectangle(img, (int(box[1]), height - int(box[2])),(int(box[3]), height - int(box[4])), (0,255,0), 2)

#cv2.imshow("img", img)
#cv2.waitKey(0)
########################

data = pytesseract.image_to_data(img, config=myconfig, output_type=Output.DICT)
amout_boxes = len(data['text'])
for i in range(amout_boxes):
    if float(data['conf'][i]) > 60:
        (x,y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(img, (x,y), (x+width, y+height), (0,255,0), 2)
        img = cv2.putText(img, data['text'][i], (x,y+height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2, cv2.LINE_AA)

cv2.imshow("img", img)
cv2.waitKey(0)



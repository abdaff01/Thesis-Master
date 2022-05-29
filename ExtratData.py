print('Libraries importation')
# Import libraries
from PIL import Image
import pytesseract
import sys
from pdf2image import convert_from_path
import os
import cv2
from matplotlib import pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.data.path.append("/home/abdelhay")
import gensim
import string
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize

#Function for filters on Images
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    #Binarization
    gray_image = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
    #grayscale
    thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)

    return(im_bw)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

# Path of the pdf
print('read PDF')
PDF_file = "szakdolgozat2.pdf"

'''
Part #1 : Converting PDF to images
'''

print('convert PDF to Images')

# Store the first 10 pages of the PDF in a variable
all_pages = convert_from_path(PDF_file, 300, first_page=0, last_page=9)

# Counter to store images of each page of PDF to image
image_counter = 1

# Iterate through all the pages stored above
for page in all_pages:
    # Declaring filename for each page of PDF as JPG
    # For each page, filename will be:
    # PDF page n -> page_n.jpg
    filename = "page2_" + str(image_counter) + ".jpg"

    # Save the image of the page in system
    page.save(filename, 'JPEG')

    # Increment the counter to update filename
    image_counter = image_counter + 1

'''
Part #2 - Applying filters on images
'''

print('Applying filters on the extracted Images')

# Create Tesseract configuration
myconfig = r"--psm 12 --oem 3"

# Variable to get count of total number of pages
file_limit = image_counter - 1

# Iterate from 1 to total number of pages
for i in range(1, file_limit + 1):
    # Set filename to recognize text from
    # Again, these files will be:
    # page_n.jpg
    filename = "page2_" + str(i) + ".jpg"
    filter1 = display(filename)
    img2 = cv2.imread("page2_" + str(i) + ".jpg")
    img = cv2.cvtColor(filter1, cv2.COLOR_BGR2RGB)
    filter2 = noise_removal(img)
    height, width, _ = filter2.shape

    boxes = pytesseract.image_to_boxes(filter2, config=myconfig)
    for box in boxes.splitlines():
        box = box.split(" ")
        img = cv2.rectangle(filter2, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0),
                            2)
    img3 = cv2.resize(filter2, (width // 4, height // 4))
    cv2.imshow('', img3)
    cv2.waitKey(0)

'''
Part #3 - Recognizing text from the images using OCR
'''

print('extract texts from Images')
# Creating a text file to write the output
outfile = "out_text2.txt"

# Open the file in append mode so that
# All contents of all images are added to the same file
f = open(outfile, "a")

# Iterate from 1 to total number of pages
for i in range(1, file_limit + 1):
    # Set filename to recognize text from
    # Again, these files will be:
    # page_n.jpg
    filename = "page2_" + str(i) + ".jpg"

    # Recognize the text as string in image using pytesserct
    text = str(((pytesseract.image_to_string(Image.open(filename)))))
    text = text.replace('-\n', '')

    # Finally, write the processed text to the file.
    f.write(text)

# Close the file after writing all the text.
f.close()

'''
Part #4 - Analyzing the text file
'''

print('analyzing the text file')

# Creating a text file to write the output
analize = "analize.txt"

# Open the file in append mode so that
# All contents of all images are added to the same file
fa = open(analize, "a")
with open("out_text2.txt") as text_file:
    text1 = text_file.read()
tokens = word_tokenize(text1)
lowercase_tokens = [t.lower() for t in tokens]

# print(lowercase_tokens)
bagofwords_1 = Counter(lowercase_tokens)

# print(bagofwords_1.most_common(10))
alphabets = [t for t in lowercase_tokens if t.isalpha()]

words = stopwords.words("hungarian")
stopwords_removed = [t for t in alphabets if t not in words]

# print(stopwords_removed)

lemmatizer1 = WordNetLemmatizer()
lem_tokens = [lemmatizer1.lemmatize(t) for t in stopwords_removed]
bag_words = Counter(lem_tokens)
resu = str(bag_words.most_common(10))

# Finally, write the processed text to the file.
open('analize.txt', 'w').close()
fa.seek(0)
fa.write(resu)
fa.truncate()

print('done')

import os
import json
import numpy as np
import cv2
import datetime
import imutils
import urllib
import csv
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", default=0, help = "debug file")
ap.add_argument("-w", "--watch", default=0, help = "Toggle imshow of processing")
args = vars(ap.parse_args())

#camera frame url
with open('keyfile.txt', 'r') as keyfile:
    frame_url = keyfile.read()
##########################################
#define Cascade Detector
##########################################
model_file = 'haarcascade_upperBody.xml'
model = cv2.CascadeClassifier(model_file)
##########################################
#define funciton to encode image from url
##########################################
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
##########################################
#define HOG detector
##########################################
#set HOG parameters
winStride = (2,2)
padding = (4,4)
meanShift = False
#initialize HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
##########################################
#setup additional parameters
##########################################
#initial frame for comparison
previousFrame = None
#data log directory
data_directory = './data_logs/'

while(True):
    ###############################################
    ##Prepare images from url stream
    ###############################################
    #set frame timestamp
    frameTimestamp = str(datetime.datetime.now())
    #capture frame by frame
    frame = url_to_image(frame_url)
    #resize
    frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
    #convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #GaussianBlur
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    Gaussian = cv2.GaussianBlur(frame, (21,21), 0)
    ###############################################
    ##Detect and Identify Movement in Frame
    ##and log data to csv file
    ###############################################
    #setup previous frame for delta calc
    if previousFrame is None:
        previousFrame = gray
        continue
    #compute difference
    frameDelta = cv2.absdiff(previousFrame, gray)
    thresh = cv2.threshold(frameDelta, 25,255,cv2.THRESH_BINARY)[1]
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(thresh,kernel,iterations = 2)
    #findContours
    contours = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    #if people are detected,
    #count them and log paratmers
    if len(contours)>0:
        for c in contours:
            area = cv2.contourArea(c)
            (enc_x,enc_y), radius = cv2.minEnclosingCircle(c)
            center = (int(enc_x),int(enc_y))
            if area < 5000.0 and area > 200.0 and enc_y>120 and enc_x >25:
                #continue
                cv2.drawContours(Gaussian, [c], -1, (0,255,0), 3)
                cv2.circle(Gaussian, center, int(radius), (255,0,0), 2)
                frame_data = {'timestamp':frameTimestamp,'location_x':int(enc_x),'location_y':int(enc_y),'contour_area':area,'enclosing_radius':radius}

                #save data
                with open(r'{}hourglass_movement_data.csv'.format(data_directory), 'a', newline='') as csvfile:
                    fieldnames = ['timestamp','location_x','location_y','contour_area','enclosing_radius']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(frame_data)
    #if people are not detected
    #log empty timeframe
    else:
        frame_data = {'timestamp':frameTimestamp,'location_x':np.nan,'location_y':np.nan,'contour_area':np.nan,'enclosing_radius':np.nan}
        #save data
        with open(r'{}hourglass_movement_data.csv'.format(data_directory), 'a', newline='') as csvfile:
            fieldnames = ['timestamp','location_x','location_y','contour_area','enclosing_radius']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(frame_data)
    ###############################################
    #Cascade detector
    ###############################################
    persons = model.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=2, minSize=(2,2))
    #if people are detected,
    #count them and log paratmers
    if len(persons)>0:
        for (x,y,w,h) in persons:
            cv2.rectangle(Gaussian, (x,y), (x+w,y+h), (255,0,0),2)
            width = ((x+w)-x)
            height = ((y+h)-y)
            area = width*height
            center_x = (x+(width/2))
            center_y = (y+(height/2))
            frame_data = {'timestamp':frameTimestamp,'location_x':int(center_x),'location_y':int(center_y),'contour_area':area}
            #save data
            with open(r'{}hourglass_cascade_data.csv'.format(data_directory), 'a', newline='') as csvfile:
                fieldnames = ['timestamp','location_x','location_y','contour_area']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(frame_data)
    #if people are not detected
    #log empty timeframe
    else:
        frame_data = {'timestamp':frameTimestamp,'location_x':np.nan,'location_y':np.nan,'contour_area':np.nan}
        with open(r'{}hourglass_cascade_data.csv'.format(data_directory), 'a', newline='') as csvfile:
            fieldnames = ['timestamp','location_x','location_y','contour_area']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(frame_data)
    ###############################################
    #HOG Detector
    ###############################################
    (HOG_rects, HOG_weights) = hog.detectMultiScale(gray, winStride=winStride, padding=padding, scale=1.01, useMeanshiftGrouping=meanShift)
    #if people are detected,
    #count them and log paratmers
    if len(HOG_rects)>0:
        for (x,y,w,h) in HOG_rects:
            cv2.rectangle(Gaussian,(x,y),(x+w,y+h),(0,0,255),2)
            width = ((x+w)-x)
            height = ((y+h)-y)
            area = width*height
            center_x = (x+(width/2))
            center_y = (y+(height/2))
            frame_data = {'timestamp':frameTimestamp,'location_x':int(center_x),'location_y':int(center_y),'contour_area':area}
            #save data
            with open(r'{}hourglass_HOG_data.csv'.format(data_directory), 'a', newline='') as csvfile:
                fieldnames = ['timestamp','location_x','location_y','contour_area']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(frame_data)
    #if people are not detected
    #log empty timeframe
    else:
        frame_data = {'timestamp':frameTimestamp,'location_x':np.nan,'location_y':np.nan,'contour_area':np.nan}
        with open(r'{}hourglass_HOG_data.csv'.format(data_directory), 'a', newline='') as csvfile:
            fieldnames = ['timestamp','location_x','location_y','contour_area']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(frame_data)
    ###############################################
    #show frame with detected objects
    ###############################################
    if int(args['watch']) != 0:
        cv2.imshow("Detections",Gaussian)
        #setup quit function key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #save existing frame as previous frame
    previousFrame = gray

    if int(args['debug']) != 0:
        print(frameTimestamp)
print('--------------------------------- Project Stream Interrupted---------------------------------')
print('---------------------------------------------------------------------------------------------')
cv2.destroyAllWindows()

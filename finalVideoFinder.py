## ***************************************************************************
##      Program to apply the HOG transformation in the input pictures
##          Coded by:
##              Marcos Ahuizotl "Ahui" Fragoso Iñiguez
##          Date:
##              March the 1, 2017
##****************************************************************************

import time
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
import pickle
import imageio
from scipy.ndimage.measurements import label
#Libraries for my SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
clf = joblib.load('carClassColor.pkl')
hogClf = joblib.load('carClassHOG.pkl')
#****************
spatial = 32
histbin = 32
colorspace = 'YCrCb'# Can be RGB, HSV, LUV, HLS, YUV
#*****HOG********
orient = 9
pix_per_cell = 8
cell_per_block = 8
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

#Picke Data
calibrator = "calibrationFiles.p"
with open(calibrator, mode='rb') as f:
    calibra = pickle.load(f)

mtx, dist = calibra['mtx'],calibra['dist']

clip1 = imageio.get_reader("project_video.mp4", 'ffmpeg')
fps = clip1.get_meta_data()['fps']
writer = imageio.get_writer('finalFantasy9.mp4', fps=fps)

##*******************************************************
##      Applied Functions
##*******************************************************
def slideWindows(img,xStartStop = [None,None],yStartStop=[None,None],xy_window =(64,64),xy_overlap=(0.5,0.5)):
    if xStartStop[0] == None:
        xStartStop[0] = 0
    if xStartStop[1] == None:
        xStartStop[1] = img.shape[1]
    if yStartStop[0] == None:
        yStartStop[0] = 0
    if yStartStop[1] == None:
        yStartStop[1] = img.shape[0]
    #Compute the span of he region to be searched
    xspan = xStartStop[1]-xStartStop[0]
    yspan = yStartStop[1]-yStartStop[0]
    #Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))    
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-nx_buffer)/ny_pix_per_step) 
    if ny_windows == 0:
        ny_windows = 1
    # Initialize a list to append window positions to
    window_list = []
    mado_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + xStartStop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + yStartStop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
            cropped = img[starty:endy, startx:endx]
            cropped = cv2.resize(cropped,(64,64))
            mado_list.append(cropped)
    # Return the list of windows    
    
    return mado_list, window_list


def draw_boxes(img, bboxes, color=(0, 0, 1), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def bin_spatial(img, size=(64, 64)):
    color1 = cv2.resize(img[:,:,0],size).ravel()
    color2 = cv2.resize(img[:,:,1],size).ravel()
    color3 = cv2.resize(img[:,:,2],size).ravel()
    return np.hstack((color1,color2,color3))

# Define a function to compute color histogram features  
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extractColorFeatures(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for i in range(len(imgs)):
        # Read in each one by one
        image = imgs[i]
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)                
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features

#*******HOG Functions************
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features



def extract_features(imgs, cspace='RGB', orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    features = []
    for i in range(len(imgs)):
        image = imgs[i]
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                                     orient, pix_per_cell, cell_per_block, 
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        features.append(hog_features)
    return features




def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, color =(0,0,1),thickness = 6):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6,thickness)
        centroidX = int((np.max(nonzerox)+np.min(nonzerox))/2)
        centroidY = int((np.max(nonzeroy)+np.min(nonzeroy))/2)     
    # Return the image
    return img

def findCentroid(labels):
    centroidX = 0
    centroidY = 0    
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        centroidX = int((np.max(nonzerox)+np.min(nonzerox))/2)
        centroidY = int((np.max(nonzeroy)+np.min(nonzeroy))/2)          
    return centroidX, centroidY

#----------------------------------------------------------------------
def findBordes(labels):
    centroidX = 0
    centroidY = 0
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        centroidX = np.max(nonzerox)-np.min(nonzerox)
        centroidY = np.max(nonzeroy)-np.min(nonzeroy)
    return centroidX, centroidY    

    

##*******************************************************
##      Main Program
##*******************************************************
#step1: Load the test pictures
#step2: Create a loop file to search the pictures
#step3: Divide the searching area in blocks
#step4: apply the color SVC
#step5: if a car is detected, save the position
#step6: apply the HOG SVC to confirm if I have a car
#step7: apply the heat map for a second confirmation
#step8: trace the lines
#step9: Enjoy

print("procesando....")
pos = 0
disparadorX = 115
disparadorY = 90
for im in clip1:
    #reduce my fishEyeError
    shashin = im
    shashin = cv2.undistort(shashin,mtx,dist,None,mtx)
    shashin = shashin.astype(np.float32)
    shashin =(shashin)/255
    draw_img = np.copy(shashin)
    # I create an sliding Windows search so I can check if the section its a car
    #mado are the pictures i need, ventanas are the position to trace them
    mado, ventanas = slideWindows(shashin, xStartStop=[600, 1150], 
                                 yStartStop=[400, 600],xy_window=(100,100),xy_overlap=(0.85,0.2))
    ventanaColor = []
    madoColor = []
    #here i convert my picture to color features and apply the color SVC
    colorFeatures = extractColorFeatures(mado, colorspace, spatial_size=(spatial, spatial),hist_bins=histbin)
    X = np.vstack((colorFeatures)).astype(np.float64)                        
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    for i in range(len(colorFeatures)):
        escalado = scaled_X[i].reshape(1,-1)
        hayCoche=clf.predict(escalado)
        chequea = clf.decision_function(escalado)
        if chequea>-0.5:
        #if hayCoche == 1:
            madoColor.append(mado[i])
            ventanaColor.append(ventanas[i])
    #Apply the HOG features on the selected windows
    hogFeatures = extract_features(madoColor, cspace=colorspace, orient=orient, 
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel)
    if len(madoColor)>0:
        X = np.vstack((hogFeatures)).astype(np.float64)                        
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)
        madoFinal = []
        ventanaFinal = []
        for i in range(len(hogFeatures)):
            escalado = scaled_X[i].reshape(1,-1)
            hayCoche=hogClf.predict(escalado)  
            chequea = hogClf.decision_function(escalado)
            if chequea > 0.3:            
            #if hayCoche == 1:
                madoFinal.append(madoColor[i])
                ventanaFinal.append(ventanaColor[i])   
            
    ## ********************************************************************
    mado, ventanas = slideWindows(shashin, xStartStop=[680, None], 
                                yStartStop=[380,520], 
                                xy_window=(140,140),xy_overlap=(0.75,0.0))
    ventanaColor = []
    madoColor = []
    #here i convert my picture to color features and apply the color SVC
    colorFeatures = extractColorFeatures(mado, colorspace, spatial_size=(spatial, spatial),hist_bins=histbin)
    X = np.vstack((colorFeatures)).astype(np.float64)                        
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    for i in range(len(colorFeatures)):
        escalado = scaled_X[i].reshape(1,-1)
        hayCoche=clf.predict(escalado)
        chequea = clf.decision_function(escalado)
        if chequea > -0.4:        
        #if hayCoche == 1:
            madoColor.append(mado[i])
            ventanaColor.append(ventanas[i])
    #Apply the HOG features on the selected windows
    hogFeatures = extract_features(madoColor, cspace=colorspace, orient=orient, 
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel)
    if len(madoColor)>0:
        X = np.vstack((hogFeatures)).astype(np.float64)                        
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)
        for i in range(len(hogFeatures)):
            escalado = scaled_X[i].reshape(1,-1)
            hayCoche=hogClf.predict(escalado)    
            chequea = hogClf.decision_function(escalado)
            if chequea > 0.4:            
            #if hayCoche == 1:
                madoFinal.append(madoColor[i])
                ventanaFinal.append(ventanaColor[i])
                
     
            

    draw_img = np.copy(shashin)
    #draw_img = draw_boxes(shashin, ventanaFinal,color=(1,1,0))
    #Here I make my heat maps
    heat = np.zeros_like(shashin[:,:,0]).astype(np.float)
    heat = add_heat(heat,ventanaFinal)
    heat = apply_threshold(heat,1) 
    heatmap = np.clip(heat, 0, 1)
    labels = label(heatmap)
    # here i have a function I use to measure the centroid of the picture and check how much it moves
    #draw_img = findCentroid(draw_img, labels)
    #aqui yo uiero guardar los labels previos y los labels actuales
    
    if pos > 0:
        centX, centY = findBordes(labels)
        centXpre, centYpre = findBordes(labelsPre)
    else:
        centX = 0
        centY = 0
        labelsPre = labels    
    
    #here I dray my output
    if centX > disparadorX and centY > disparadorY:
        draw_img = draw_labeled_bboxes(draw_img, labels,color=(0,1,0),thickness=6)
        print("si que si", pos)
        
    else:
        #draw_img = draw_labeled_bboxes(draw_img, labels,color=(0,1,0),thickness=1)
        draw_img = np.copy(shashin)
    #plt.imshow(draw_img)
    #plt.show()
    print(centX,centY)
    writer.append_data(draw_img)
    pos = pos + 1
print("listo!!!!")
  
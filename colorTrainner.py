## ***************************************************************************
##      Program to apply the HOG transformation in the input pictures
##          Coded by:
##              Marcos Ahuizotl "Ahui" Fragoso Iñiguez
##          Date:
##              March the 1, 2017
##****************************************************************************

#load the required libraries
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import random
import csv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

##*******************************************************
##      Applied Functions
##*******************************************************
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
def extract_features(imgs, cspace='LUV', spatial_size=(32, 32),
                        hist_bins=32):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
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


##*******************************************************
##      Main Program
##*******************************************************
#Step 1: load all the required pictures for the trainning
cars = glob.glob('vehicles/*/*.png')
print("total of car pictures: ",len(cars))

carsBeta = glob.glob('vehiclesBeta/*.png')

notcars = glob.glob('non-vehicles/*/*.png')

notcarsBeta = glob.glob('nonVehiclesBeta/*.png')
print("total of non vehicles pictures: ",len(notcars))

testCar = random.randint(0, len(cars))
testNoCar = random.randint(0, len(notcars))

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
sample_size = 4000
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]
    
for i in range(len(notcarsBeta)):
    notcars.append(notcarsBeta[i])
    cars.append(carsBeta[i])


print("total of car pictures: ",len(cars))
print("total of non vehicles pictures: ",len(notcars))
spatial = 32
histbin = 32
colorspace = 'YCrCb'# Can be RGB, HSV, LUV, HLS, YUV

cars, notcars = shuffle(cars,notcars)

car_features = extract_features(cars, colorspace, spatial_size=(spatial, spatial),
                                hist_bins=histbin)
notcar_features = extract_features(notcars, colorspace, spatial_size=(spatial, spatial),
                                   hist_bins=histbin)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), -1*np.ones(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state)

print('Using spatial binning of:',spatial,
      'and', histbin,'histogram bins')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
clf = SVC(C=3,kernel='rbf',gamma='auto')
# Check the training time for the SVC
t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
from sklearn.externals import joblib
joblib.dump(clf, 'carClassColor.pkl')
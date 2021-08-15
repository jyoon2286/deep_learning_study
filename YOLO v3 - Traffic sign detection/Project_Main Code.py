import cv2
import numpy as np
import IPython
import time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

import os

#Import weight, model, and labels files for YOLO model
weight_file = 'yolov3_ts_train_5000.weights' #contains the pretrained weights for the YOLO model
cfg_file = 'traffic-sign-yolo.cfg' #contains the layers for the final YOLO model
name_file = 'classes.names'

min_confidence = 0.5

#Import test video and image files for prediction
frame_count = 0
writer = None
input_name = 'traffic-sign-video.avi'
output_name = 'traffic-sign-video.avi'
file_name = '00001.jpg'


# Load Yolo
net = cv2.dnn.readNet(weight_file, cfg_file)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Get class names from file
classes = []
with open(name_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

#Read test image file
img = cv2.imread(file_name)
height, width, channels = img.shape

# Detecting objects - convert image dimensions and feed into model
# https://docs.opencv.org/master/d6/d0f/group__dnn.html
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

#Retrieve bounding box dimensions from model output
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


###Begin creating and training classification CNN model

#Import training images for classification model
with open('./data2.pickle', 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # dictionary type

# Preparing train, validation, and test sets for classification model
data['y_train'] = to_categorical(data['y_train'], num_classes=43)
data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)
data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

# Print shape of data sets
for i, j in data.items():
    if i == 'labels':
        print(i + ':', len(j))
    else: 
        print(i + ':', j.shape)


#Create function for visualizing training data
def convert_to_grid(x_input):
    N, H, W, C = x_input.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + 1 * (grid_size - 1)
    grid_width = W * grid_size + 1 * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C)) + 255
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = x_input[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)
                next_idx += 1
            x0 += W + 1
            x1 += W + 1
        y0 += H + 1
        y1 += H + 1

    return grid


# Visualizing some examples of training data
examples = data['x_train'][:81, :, :, :]
print(examples.shape)  # (81, 32, 32, 3)

fig = plt.figure()
grid = convert_to_grid(examples)
plt.imshow(grid.astype('uint8'), cmap='gray')
plt.axis('off')
plt.gcf().set_size_inches(15, 15)
plt.title('Some examples of training data', fontsize=18)

plt.show()

# Saving the plot
fig.savefig('training_examples.png')
plt.close()


#Build CNN model for classification into 43 traffic sign classes
class_model = Sequential()
class_model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(32, 32, 3)))
class_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
class_model.add(Conv2D(filters=96, kernel_size=(5,5), padding='same', activation='relu', input_shape=(32, 32, 3)))
class_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
class_model.add(Flatten())
class_model.add(Dense(1000, activation='relu'))
class_model.add(Dropout(0.4))
class_model.add(Dense(43, activation='softmax'))
class_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
class_model.summary()

#Fit classification model on training and validation data
batch = 100
epochs = 15
class_model.fit(data['x_train'], data['y_train'],
                    batch_size=batch, epochs = epochs,
                    validation_data = (data['x_validation'], data['y_validation']), verbose=1)


#Predict on test data and print testing accuracy of model
tmp = class_model.predict(data['x_test'])
tmp = np.argmax(tmp, axis = 1)
accuracy = np.mean(tmp == data['y_test'])
print("Batch Size = ", str(batch), "; Epochs = ", str(epochs), "; Test Accuracy: ", accuracy)


#Predict bounding boxes and labels on inputted test image and visualize
indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_COMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        #Slice original image to only the bounding boxes, then redimension
        image_box = img[y:y+h,x:x+w,:]
        photo_box = cv2.dnn.blobFromImage(image_box, 0.00392, (32, 32), (0, 0, 0), True, crop=False)
        photo_box = photo_box.transpose(0, 2, 3, 1)
        #Run classification model on the bounding-box images
        scores = class_model.predict(photo_box)
        photo_pred = np.argmax(scores)
        labels_tbl = pd.read_csv('label_names.csv')
        sign_class = labels_tbl[labels_tbl['ClassId'] == photo_pred]['SignName'].iloc[0]
        print('Predicted sign class of bounding box ' + str(i+1) + ' is: ' + sign_class + '.')
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, sign_class, (x, y - 10), font, 0.5, (0, 255, 0), 1)

#Visualize the predicted bounding boxes and labels
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()





# The following section displays our implementation of prediction on a video file. The following code is able to run on the Google Collab environment
# but does not run in an offline environment. Therefore, we are not including it as part of our submission. The output file of this code
# that includes the predictions from runs in the Google environment is, however, included in the zip folder as file 'YOLO_traffic_output.mp4'



# # #2. Video Detection - Run YOLO and classification models on video file

# #Function to write predictions back to output video file
# def writeFrame(img):
#     # use global variable, writer
#     global writer
#     height, width = img.shape[:2]
#     if writer is None and output_name is not None:
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#         writer = cv2.VideoWriter(output_name, fourcc, 24, (width, height), True)
#     if writer is not None:
#         writer.write(img)


# #Function to run model on frame of video file
# def detectAndDisplay(frame):
#     # use global variable, writer
#     global frame_count
#     frame_count += 1
#     start_time = time.time()
#     IPython.display.clear_output(wait=True)
#     height, width, channedls = frame.shape


#     # Detecting objects - convert image dimensions and feed into YOLO model
#     # https://docs.opencv.org/master/d6/d0f/group__dnn.html
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     class_ids = []
#     confidences = []
#     boxes = []

#     #Retrieve bounding box dimensions from model output
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > min_confidence:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     #Slice full-frame to only bounding boxes and run classification on bounding-box images
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
#     font = cv2.FONT_HERSHEY_COMPLEX
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             image_box = frame[y:y+h,x:x+w,:]
#             frame_box = cv2.dnn.blobFromImage(image_box, 0.00392, (32, 32), (0, 0, 0), True, crop=False)
#             frame_box = frame_box.transpose(0, 2, 3, 1)
#             scores = class_model.predict(frame_box)
#             photo_pred = np.argmax(scores)
#             sign_class = labels_tbl[labels_tbl['ClassId'] == photo_pred]['SignName'].iloc[0]
#             print('Predicted sign class of bounding box ' + str(i+1) + ' is: ' + sign_class + '.')
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, sign_class, (x, y - 10), font, 0.5, (0, 255, 0), 1)  

#     #Visualize frames with predictions and save back to file
#     frame_time = time.time() - start_time 
#     print("Frame {} time {}".format(frame_count, frame_time))
#     cv2.imshow("Video Window",frame)   
#     writeFrame(frame)


# #-- 2. Read the video stream
# cap = cv2.VideoCapture(input_name)
# if not cap.isOpened:
#     print('--(!)Error opening video capture')
#     exit(0)
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print('--(!) No captured frame -- Break!')
#         break
#     detectAndDisplay(frame)
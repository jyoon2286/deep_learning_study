# Name - Project Report<br>
Traffic Sign Classification with YOLO v3 Framework<br>
By: Jaehyun Yoon, Renea Young, Andrew Tran

# Background
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Object detection is a computer vision technique that identifies and locates objects in an
image or video. The identification and location of objects can be done by creating bounding
boxes around the objects and labeling them. Object detection can be applied in a number of
ways. Crowd counting, video surveillance, face recognition, and self-driving are some of the
primary ways object detection can be applied.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Traffic sign detection is one of the essential technologies in the field of assisting drivers
such as Advanced driver-assistance systems (ADAS), transportation systems, and autonomous
driving. The world of neural networks and autonomous vehicles is evolving fast. Apple,
Volkswagen, Honda, and other companies are relying on artificial intelligence to improve car
safety. Traffic sign detection is a challenging task in which the algorithms have to cope with
natural and complex environments, high accuracy demands, and real-time constraints. The
challenges that arise when detecting traffic signs are blur due to capturing the image from a
moving vehicle, displacement of the traffic symbols, faded traffic symbols due to the effect of
weather elements, and many more. Many approaches have been proposed for traffic sign
detection. This study focuses on using the You Only Look Once version 3 (YOLOv3) framework
to detect the German traffic signs detection benchmark (GTSDB) database.


# The Data
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For our project of traffic sign detection, we are using data sets compiled by the Institute
Fur Neuroinformatik based in Germany. The data sets used in this study are specifically provided
by the group for the purpose of solving computer vision and machine learning problems. The two
data sets they provide are the German Traffic Sign Detection Benchmark Set (“GTSDB”) and the
German Traffic Sign Recognition Benchmark (“GTSRB”) set. The GTSDB set is composed of
900 images of landscape photos, each of which includes some aspect of traffic signs in different
scenes to be used for solving object-detection problems. The GTSRB is composed of over
50,000 images of zoomed-in traffic signs to be used for solving classification and recognition
problems. The traffic signs within both of these sets contribute to a total of 43 different traffic
sign classes. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The GTSDB data set was provided in a folder of about 2.9GB in size, with each image in
a .ppm format containing the image details in a text format. The data in the .ppm file represents
the images in their original dimensions, therefore, the data will need to be reformatted before
running the model. Each image is also accompanied by a separate .txt file that contains the label
details for the class of each traffic sign in the image, as well as the coordinates of where the
traffic signs are located within the image for training and testing purposes. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The GTSRB data set was provided in a .pickle file of about 1.3GB in size containing a
binary encoding of the image pixel details as well as their labels corresponding to the 43 classes.
This .pickle file also includes a preset 83%-training, 4%-validation, and 13%-test split on the
data that we can utilize for testing our model. The .pickle file holds preprocessed data of the
images where they have been normalized and standardized so the pixel values are within the
range of 0 to 255.

# Implementation
####  YOLO Implementations
####  Preparing the Data
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The YOLO requires the user to provide the input in the YOLO format, which is in the
order of class id, the center X, the center Y, and height, and width for each image. The dataset
provides a Roi of X and a Roi of Y for each object. We need to calculate the center X, the center
Y, height, and width based on the provided Roi. We customize the configuration file for our
model in which we need to set the number of batches, the max batch, number of classes, and
filters. For the max batch, Darknet recommends setting the max batch as the number of classes
times 2000. Since we have limited GPU and CPU to train our weights, it wasn’t feasible to train
the weights with all 43 classes in our local machine. We decide and divide into 4 different classes
which are “Danger”, “Prohibited”, “Mandatory”, and “other”. With only the 4 classes of weights,
it took us 6 to 8 hours to get the weights in the cloud machine. Darknet provides a function to
save the best weight file and the backup files for every 1000 epochs in case of getting an error
and being stopped during the training.<br>

#### Building the Model and Predicting with the Model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The graph shows how the loss value and the mean average precision (mAP) changed as
we train the weights. As you can see in the Figure 6, the result of loss value dropped right about
the 400 epochs and the mean average precision showing at about the 1000 epochs. Since we have
max batches as the 8000 (4 classes *2000), it will run until 8000 epochs. We decided to stop at
the 5000 epochs instead of all the way to 8000 epochs to prevent overfitting. The best mean
average precision (mAP) was about 95.35 percent and the loss value got to close to zero which
was 0.000010.

#### Classification CNN Model Implementation
#### Preparing the Data
For classification, we used the which we downloaded in a .pickle file of binary encoding.
To prepare the data for training the classification model, we needed to import the data into a
usable array format by importing the “pickle” package and using its “load” function with and
specifying the “encoding” parameter to reformat the data. Then, we must pull out the
predetermined training, validation, and test data split designed in the source data. We must also
encode the target variables into categorical labels set to the desired 43 classes to fit with our
classification model. Lastly, we need to redimension the image data so that the
channels-dimension (for the three color channels) are the last dimension of the input data. After
completing all of these steps, the data is now ready to be fed into the classification model for
training. <br>
#### Building the Model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In building the image classification model, we first started out with using the popular
AlexNet architecture that is known for being able to efficiently and accurately predict labels of
image data, such as the well-known MNIST handwritten-digits and flower-iris data sets. AlexNet
is known for having eight main layers: five convolutional layers and three fully connected layers.
Despite the architecture’s strong reputation, we found that AlexNet yielded poor accuracy on the
test data set, sitting at around 55% accuracy. Interestingly, in further tests, we gradually found
that removing layers from AlexNet started to improve accuracy. After testing multiple
architectures, we decided on implementing an eight-layer architecture with only two
convolutional layers, two max pooling layers, and two fully connected layers. This architecture
looked to yield the best training and testing accuracy on this data set. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After deciding on a single architecture, we then trained this on the GTSRB dataset with
the preset training, validation, and test split of 83%, 4%, and 13%, respectively. With this final
architecture, we also tested different batch sizes and epoch parameters to find what combination
maximized testing accuracy on the GTSRB data set. In these tests, we tested batch sizes of 5, 10,
50, and 100, and tested epochs of 3, 5, 10, and 15. In each of these tests, we tracked the total time
needed to train the model, the training accuracy, the validation accuracy, and the test accuracy for
comparison purposes. The metrics found from each test can be seen in the table below:


# Example
<p align="center">
  <img  width="600" height="400" src=dataset/00001.jpg>
</p>

<p align="center">
  <img  width="600" height="400" src=dataset/yolo_output.png>
</p>

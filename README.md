# Automatic-Weapon-Detection
deployment of a hybrid model for automatic weapon detection/ anomaly detection for surveillance applications


## Project Summary:
In this project CNN algorithm is simulated for pre-labelled image dataset for weapon (gun, knife) detection. The algorithm is efficient and gives good results but its application in real time is based on a trade-off between speed and accuracy. With respect to accuracy, CNN gives accuracy of approx. 85%. In our CNN model we have taken 16 layers. Apart from this the optimiser used by us is SGD, with categorical cross entropy loss and accuracy is used as the metrics. For every layer we have used the ‘relu’ activation function, for the last layer we have used softmax. We have used Tensorflow, Keras, PIL, OpenCV, Playsound modules to implement the project. Our software takes a 240 x 240 image as input, in a batch size of 32.

Further, it can be implemented for larger datasets by training using GPUs and high-end DSP and FPGA kits.

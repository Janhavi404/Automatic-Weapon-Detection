# Automatic-Weapon-Detection
Deployment of a hybrid model for automatic weapon detection/ anomaly detection for surveillance applications

## Features to Detect Weapons / Intruders
### Knives: 
We propose algorithms that are able to alert the human operator when a firearm or knife is visible in the image. We have focused on limiting the number of false alarms in order to allow for a real-life application of the system. The specificity and sensitivity of the knife detection are significantly better than others published recently. We have also managed to propose a version of a firearm detection algorithm that offers a near-zero rate of false alarms. We have shown that it is possible to create a system that is capable of an early warning in a dangerous situation, which may lead to faster and more effective response times and a reduction in the number of potential victims.

### Size:
Estimation of the size of software is an essential part of Software Project Management. It helps the project manager to further predict the effort and time which will be needed to build the project. Various measures are used in project size estimation. Some of these are:
•	Lines of Code
•	Number of entities in ER diagram
•	Total number of processes in detailed data flow diagram
•	Function points

Find the number of functions belonging to the following types:
•	External Inputs: Functions related to data entering the system.
•	External outputs: Functions related to data exiting the system.
•	External Inquiries: They leads to data retrieval from system but don’t change the system.
•	Internal Files: Logical files maintained within the system. Log files are not included here.
•	External interface Files: These are logical files for other applications which are used by our system.



### Trigger:
Detecting small objects is a difficult task as these objects are rather smaller than the human. In this section, we will implement a gun detector that trained by using the discriminatively trained part-based models. As our object of interest is gun, we will collect different positive samples from different type of gun related videos. To minimize the amount of supervision, we provide the bounding box of the gun in the first frame where the gun appears and apply the tracking method to let it track for the gun. We will then use the result from the tracker to annotate the gun location in each image. For the negative samples, we will use all the annotation from the Pascal Visual Object Classes Challenge (VOC) as all the annotations are without any gun object. Lastly, all the annotation results of the positive sample and negative samples are used as the input for the DPM to train a gun model. Tracking is required in different stages of our system because the object detector tends to produce sparse detection as the object of interest is too small. 

### Handle
Cohen’s kappa coefficient is used to check the agreement between experts which is calculated using following formula:

![aaaaa](https://user-images.githubusercontent.com/65353861/119645517-bb521a00-be3b-11eb-8683-aa4e9fff1c0c.png)
 
where pa ¼ proportion of observations for agreement of two experts; pc ¼ proportion of observations for agreement which is expected to happen by chance between two experts. Agreement matrix of proportions is for weapon purchase. Cohen’ Kappa coefficient value was found to be 0.9425 at a ¼ 0.05 (a is probability of confidence interval for kappa statistics) which signifies an almost perfect agreement between the experts. R Programming Package “psych” is used to compute Cohen’s kappa coefficient. Considering significance and magnitude of kappa coefficient so computed, the annotations labelling represents the justification of process of manually labelling approach which can therefore be used in our analysis to train and test our proposed automated illegal weapon procurement model.

## Project Summary:
In this project CNN algorithm is simulated for pre-labelled image dataset for weapon (gun, knife) detection. The algorithm is efficient and gives good results but its application in real time is based on a trade-off between speed and accuracy. With respect to accuracy, CNN gives accuracy of approx. 85%. In our CNN model we have taken 16 layers. Apart from this the optimiser used by us is SGD, with categorical cross entropy loss and accuracy is used as the metrics. For every layer we have used the ‘relu’ activation function, for the last layer we have used softmax. We have used Tensorflow, Keras, PIL, OpenCV, Playsound modules to implement the project. Our software takes a 240 x 240 image as input, in a batch size of 32.

Further, it can be implemented for larger datasets by training using GPUs and high-end DSP and FPGA kits.

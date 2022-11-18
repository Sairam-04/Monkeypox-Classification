<h1> Monkeypox Classification<h1>

<h2> Abstract </h2>
  <p>
   The world is still trying to recover from the devastation caused by the widespread of COVID-19, and now the monkeypox virus threatens becoming a worldwide pandemic. Although the monkeypox virus is not as lethal or infectious as COVID-19, numerous countries report newcases daily. Thus, it is not surprising that necessary precautions have not been taken, and it will not be surprising if another worldwide pandemic occurs. 
  Recently it has been shown that machine learning is very promising for image-based diagnosis, including cancer detection, tumor cell identification, and the detection of COVID-19 patients. It is therefore possible to implement a similar application to diagnose monkeypox as it invades human skin and an image can be acquired and used to facilitate further diagnosis.
  In this project, one CNN model and three pre-trained models (MobileNet, VGG-16, ResNet-50v2) are used for the classification of monkeypox images. The image dataset used is from kaggle (Monkeypox legion dataset) with 2142 training images, 90 testing images,  253 images for validation. The objective of our classification is to determine if an image contains monkeypox or not. With a fresh image, you can upload an image and find the result of every deep learning model on our website.
  This model is supported by explainable Deep Learning methods. As a result, an artificial intelligence (AI) assisted auxiliary classification system has been proposed for Monkeypox skin lesions.
</p>
    
<h2> Dataset </h2>
<p>
There are 3 folders in the dataset.

1) Original Images: It contains a total number of 228 images, among which 102 belongs to the 'Monkeypox' class and the remaining 126 represents the 'Others' class i.e., non-monkeypox (chickenpox and measles) cases.

2) Augmented Images: To aid the classification task, several data augmentation methods such as rotation, translation, reflection, shear, hue, saturation, contrast and brightness jitter, noise, scaling etc. have been applied using MATLAB R2020a. Although this can be readily done using ImageGenerator/other image augmentors, to ensure reproducibility of the results, the augmented images are provided in this folder. Post-augmentation, the number of images increased by approximately 14-folds. The classes 'Monkeypox' and 'Others' have 1428 and 1764 images, respectively.

3) Fold1: One of the three-fold cross validation datasets. To avoid any sort of bias in training, three-fold cross validation was performed. The original images were split into training, validation and test set(s) with the approximate proportion of 70 : 10 : 20 while maintaining patient independence. According to the commonly perceived data preparation practice, only the training and validation images were augmented while the test set contained only the original images. Users have the option of using the folds directly or using the original data and employing other algorithms to augment it.

Additionally, a CSV file is provided that has 228 rows and two columns. The table contains the list of all the ImageID(s) with their corresponding label
	<br><br>
	Dataset Link : 
	<a href="https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset"> Monkeypox Skin Lesion Dataset Kaggle </a>
</p>
	
 <h2> Methodology </h2>
    
 <p>
        In Classification problems, Convolutional Neural Networks and pretrained models can be used. Therefore we propose an approach to use pretrained models and build a classifier for classifying images of monkeypox vs. non-monkeypox.
Here we build four models, one CNN and three pretrained models Resnet50V2, Mobilenet and VGG16. These trained models are saved as .h5 files where all the weights are saved of trained models and those weights are used in the web application to classify the images of monkeypox and non-monkeypox. In a web application we can choose the model out of four models to classify the images of monkeypox and non-monkeypox.

</p>
    
 <h3> Technologies Used </h3>
    <ul>
		<li> Python Programming Language </li>
		<li> Deep Learning </li>
		<li> Computer Vision </li>
	</ul>
	
<h3>Frameworks and Libraries Used</h3>
<ul>
	<li> Virtual environment </li>
	<li> OpenCV </li>
	<li> Keras </li>
	<li> Pandas </li>
	<li> Tensorflow </li>
	<li> Matplotlib </li>
	<li> Numpy </li>
	<li> Flask </li>
    
</ul>
	
<h3> Models Used </h3>
<p>In this model, we have programmed a convolutional neural network (CNN) whereas  ResNet-50v2, VGG-16, MobileNet models are existing pre-trained models.	</p>
<ul>
	
<li> Convolutional Neural Network </li><br>
<p> 
	A convolutional neural network (CNN or convnet) is a subset of machine learning. It is one of the various types of artificial neural networks which are used for different applications and data types. A CNN is a kind of network architecture for deep learning algorithms and is specifically used for image recognition and tasks that involve the processing of pixel data.
There are other types of neural networks in deep learning, but for identifying and recognizing objects, CNNs are the network architecture of choice. This makes them highly suitable for computer vision (CV) tasks and for applications where object recognition is vital, such as self-driving cars and facial recognition. The Convolutional Layer and the Pooling Layer, together form the i-th layer of a Convolutional Neural Network. Depending on the complexities in the images, the number of such layers may be increased for capturing low-level details even further, but at the cost of more computational power.
</p>

	
<li> MobileNet </li><br>
<p>
	As the name applied, the MobileNet model is designed to be used in mobile applications, and it is TensorFlow’s first mobile computer vision model.
MobileNet uses depth wise separable convolutions. It significantly reduces the number of parameters when compared to the network with regular convolutions with the same depth in the nets. This results in lightweight deep neural networks.
MobileNet is a class of CNN that was open-sourced by Google, and therefore, this gives us an excellent starting point for training our classifiers that are insanely small and insanely fast. Dense-MobileNet introduces dense block idea into MobileNet. The convolution layers with the same size of input feature maps in MobileNet model are replaced as dense blocks, and the dense connections are carried out within the dense blocks. Dense blocks can make full use of the output feature maps of the previous convolution layers, generate more feature maps with fewer convolution kernels, and realize repeated use of features. By setting a small growth rate, the parameters and computations in MobileNet models are further reduced, so that the model can be better applied to mobile devices with low memory.
</p>
	
<li> ResNet</li><br>
<p>
	ResNet stands for Residual Network. It is an innovative neural network that was first introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in their 2015 computer vision research paper titled ‘Deep Residual Learning for Image Recognition’.
ResNet has many variants that run on the same concept but have different numbers of layers. Resnet50 is used to denote the variant that can work with 50 neural network layers. Residual network architecture introduced “skip connections”. The main advantage of these models is the usage of residual layers as a building block that helps with gradient propagation during training. ResNet has many variants that run on the same concept but have different numbers of layers. Resnet50 is used to denote the variant that can work with 50 neural network layers. Resnet uses skip connections, These skip connections work in two ways. Firstly, they alleviate the issue of vanishing gradient by setting up an alternate shortcut for the gradient to pass through. In addition, they enable the model to learn an identity function. This ensures that the higher layers of the model do not perform any worse than the lower layers. In short, the residual blocks make it considerably easier for the layers to learn identity functions. As a result, ResNet improves the efficiency of deep neural networks with more neural layers while minimizing the percentage of errors. In other words, the skip connections add the outputs from previous layers to the outputs of stacked layers, making it possible to train much deeper networks than previously possible.  
 </p>
	
<li> VGG-16</li><br>
<p>
	VGG-16 is a convolutional neural network that is 16 layers deep. You can load a pre-trained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224. For more pretrained networks in MATLAB, see Pre Trained Deep Neural Networks. You can use classify to classify new images using the VGG-16 network. The VGG model, or VGGNet, that supports 16 layers is also referred to as VGG16, which is a convolutional neural network model proposed by A. Zisserman and K. Simonyan from the University of Oxford.
	The VGG network is constructed with very small convolutional filters. The VGG-16 consists of 13 convolutional layers and three fully connected layers. The number 16 in the name VGG refers to the fact that it is a 16 layers deep neural network (VGGnet). This means that VGG16 is a pretty extensive network and has a total of around 138 million parameters. Even according to modern standards, it is a huge network. However, VGGNet16 architecture’s simplicity is what makes the network more appealing. Just by looking at its architecture, it can be said that it is quite uniform. There are a few convolution layers followed by a pooling layer that reduces the height and the width. If we look at the number of filters that we can use, around 64 filters are available that we can double to about 128 and then to 256 filters. In the last layers, we can use 512 filters.

</p>
	
</ul>
	
<h3> Workflow </h3>
<p>
	The two stages of the project workflow in this instance are the web application workflow and the model training phase. In the training phase the model is trained on monkeypox skin lesion dataset. In the web application phase the web application is developed and the trained models are used in the web application to make the classifications and predictions on the given image.
</p>
	
* <h4> Model Training Phase</h4>
<p>
	The model training phase and the web application workflow are the two phases that make up this project's workflow. There are six steps in the training process. The first step is loading the dataset. Tensorflow library is used to load the monkeypox dataset from Kaggle into the Colab notebook. Image preprocessing is the second step in the training phase. In this step, numerous operations on the image are carried out, including histogram equalization, image standardization, and normalization. The model architecture is built in the third step of the training process, which is known as model building. In our study, we used four models: CNN, Resnet50V2, VGG16, and Resnet50V2. For each of these models, a model's architecture was designed.The model is assessed using test data in the fifth and final step of this process, which also involves calculating accuracy. Saving the Model is the last step in this process, where we save the model as an.h5 file. The .h5 file consists of parameters trained during the training phase of the model. The weights are saved in the .h5 file. The file is machine encoded. This saved model can also be utilized in a web application to make predictions about an image that has been provided.
</p>
	
* <h4> Workflow of Web Application <h4>
	
<p>
		The first stage in a web application workflow is to access the test page, where the user must upload an image and make a prediction based on that image. On the test page, each model is shown, and the user may choose any of them. The uploading of an image is the next stage. The third stage is choosing a model from the available models and making a prediction based on the given image. The web application uses the.h5 files, where model parameters and weights are stored, to classify the image as monkeypox or non monkeypox is the final step. The model classifies the image and generates the results using those parameters.
</p>
	
	

<h3> Application Demo </h3>
	
	
	
	
	
	

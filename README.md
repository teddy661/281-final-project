# 281-final-project

## Final Project
**DATASCI 281: Foundations of Computer Vision**

### Overview
For your final project, you will work with a small group to build a custom image classifier using the tools you have learned in this class. The type of classification problem you solve will be up to you.
First, you will need to find an image dataset online. There are many popular datasets used for classification across many domains, including medicine, architecture, agriculture, satellite imagery, etc. Your dataset should have at least 3 classes, be of at least moderate difficulty (more complicatedthan MNIST), and should already include any labels needed to answer your intended classification query (supervised learning only). You will likely need to perform some pre-processing steps on your images to ensure that they are the right size, resolution, number of channels, appropriate contrast, etc. You will turn in a proposal including information about your dataset and the intended classification question you hope to answer, along with a few other details described below.

Next, you will design custom feature vectors using the filtering and image decomposition methods covered in class. You must implement at least two simple feature types (for example: histogram of oriented gradients, HSV color histogram, local binary pattern). You may also use combinations of several simple features concatenated together. Choose feature vectors that are well-suited to the type of images and classification question you are working on. For full credit, you must also include at least one complex feature from a library or pre-trained neural network (for example: ResNet, VGG).
Be sure to apply any necessary pre-processing and use parameters appropriate for your dataset.

Finally, you will build and train at least two classifiers using two different classification methods that we have learned (for example: logistic regression, SVM, simple perceptron). You should (again) use classification methods that are appropriate for your dataset and query. Consider working with a smaller subset of the data to test out your ideas for classifier design. During this step, you should divide up your dataset into training, validation, and testing groups. For full credit, implement a hyperparameter search using subsets of the validation data. Avoid overfitting.
The final deliverable will be a link to your well-commented code, and a written report (PDF) describing your process, with figures showing the results for each step. To ensure that your projects are on-track, each group will provide short weekly status updates in the second half of the semester, including a final presentation during the last week of class.

### Rubric
**Total** 100 pts</br>
**Proposal** - 5 pts</br>
Provide a link to the dataset, example images, description of the variation in the dataset (e.g. categories, size/resolution, etc), description of the intended classification problem (i.e., list of output categories), estimation of the approximate number of images expected to be in each category, and a guess of the types of image features that may be useful for this categorization (e.g. edges, histograms, etc)</br></br>
**Project Description**</br>
Classification of ... </br>
[Dataset link](https://www.kaggle.com/code/ryanholbrook/create-your-first-submission/notebook)</br>
**Example Images**</br></br>

**Feature Extraction** - 35 pts</br>
This part should include code to extract features, illustrations of the features extracted from several example images, plots showing the amount of variation in the dataset, as well as PCA decomposition and tSNE visualization of features. Be sure to accurately describe and interpret your methods and results. You must include at least two simple features and at least one complex feature (see above). Include a detailed explanation of why you chose these particular features for your dataset and classification problem.</br>
**Classification** - 35 pts</br>
This part should include code to perform classification using at least two methods learned in class, plots showing the results of classification per category, a discussion of possible reasons why the classifier might work better for some categories than others, and explanation of the limitations of the classifier.</br>
**Generalizability** - 10 pts</br>
Your data should be split into train, validation, and test groups before training the classifier, and you should do a hyperparameter search using parts of the validation set in a way that avoids over-fitting and maximizes generalizability. Report performance on the test set, and include a discussion of whether you achieved generalizability, and how your training process might be improved.</br>
**Efficiency vs Accuracy** - 10 pts</br>
For the various combinations of three feature vectors and two classifiers, include characterization of both accuracy and training/inference cost (based on time, assuming equal computational power). Optimize one solution for accuracy and one for efficiency. Include a discussion of the relative trade-offs necessary for each of these solutions.</br>
**Quality of Explanation** - 5 pts</br>
Overall quality of report, including readability of figures and code comments, quality of analysis, and discussion of limitations.</br>

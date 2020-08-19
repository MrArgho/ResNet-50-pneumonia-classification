# ResNet-50-pneumonia-classification

Built ResNet50 model using TensorFlow Keras to do the classification of pneumonia images (whether the X-ray image shows normal or pneumonia)

Data source from Kaggle:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Original ResNet50 paper reference:
https://arxiv.org/abs/1512.03385

ResNet50 architecture as below:

Identity block:

Convolution block:

The original image data set from Kaggle has 3 folders; train, val and test. However, there are only 8 images in the val folder. So I combined the train and val folder into one folder trainval and split the whole dataset into train and val by 8:2 using sklearn train-test-split.

As I train the model on my own laptop (MacBook Pro 15-inch 2015) using CPU, it took about 10 hours? (As I went to sleep when it was training, so not sure the exact time). 

The model achieved 99.69% accuracy on training set and 76.79% accuracy on validation set.


We can see that there is overfitting problem on the training data.
To do list:
- Fine tune the parameters
- Retrain th model using Cloud computing
- Try using pretrained model in Keras libray

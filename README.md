# Face Recognition

## Abstract 

This project is the first project completed for Machine Learning and Data Mining course during my senior year at the University of Toronto. In this project, I explored the effectiveness of applying linear regression on classfications problems of two different actors, two different genders and the overfitting problems. I also visualized the final weights used on the feature vectors. 

## Problems 

The specific instructions of this project could be found in the [instructions](./Instructions.pdf) file in this repository. I first downloaded the [!FaceScrub](http://vintage.winklerbros.net/facescrub.html) dataset from the UC Irvine machine learning repository. The original images looks like the following. 

[![Capture.png](https://s14.postimg.org/ymfpzaq7l/Capture.png)](https://postimg.org/image/io7095vzh/)

There was a large amount of the data cleaning required, and they were completed using the ***update_images*** function in the ***faces.py*** files. After the preprocessing the images are all cropped and reshaped to 32 X 32 in gray scale. The example images are shown below. 

[![Capture.png](https://s18.postimg.org/ok59iyp7d/Capture.png)](https://postimg.org/image/bsr3cgff9/)

Even though there are some mislabelled images, they will not affect the result that much since the dataset is relatively large.

I used the linear regression classify to explore the following four problems

1. classify whether the given image is a male actor or a female actress
2. classify whether the given image is Bill Hader or Steven Carell
3. visualize the weights of the features as images
4. explain the effect of overfitting

## Result

Surprisingly, the linear regression classifier could perform relatively well to distinguish male from female, as well as distinguish Bill from Steven. I believe the data happen to be quite separable for the linear classifier to return a strong performance. The visualized weights are shown below

[![Capture.png](https://s14.postimg.org/hn6rk6k6p/Capture.png)](https://postimg.org/image/nbd2b2oj1/)

With less images in the training set, we obtain a relatively more clear image of a face.

The plot of training/test set performance is shown below

[![Capture.png](https://s14.postimg.org/82n4xa7pd/Capture.png)](https://postimg.org/image/469t1ampp/)

It is quite evident that the accuracy of the test set stopped increasing after a certain number of iterations, which could be an early signal of diminishing returns of further training epoches.


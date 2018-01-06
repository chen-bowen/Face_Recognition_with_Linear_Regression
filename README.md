# Face Recognition

## Abstract 

This project is the first project completed for Machine Learning and Data Mining course during my senior year at the University of Toronto. In this project, my fellow teammate, Yuan Yao, and me explored the effectiveness of applying linear regression on classfications problems of two different actors, two different genders and the overfitting problems. We also visualized the final weights used on the feature vectors. 

## Acknowledgements

I would like to thank my collaborator, Yuan Yao, for his strong dedication and intelligence to make this project a great one. 

## Problems 

The specific instructions of this project could be found in the [instructions](./Instructions.pdf) file in this repository. We first downloaded the [!FaceScrub](http://vintage.winklerbros.net/facescrub.html) dataset from the UC Irvine machine learning repository. There was a large amount of the data cleaning required, and they were completed using the ***update_images*** function in the ***faces.py*** files. After the preprocessing the images are all cropped and reshaped to 32 X 32 in gray scale. The example images are shown below. 

[![Capture.png](https://s18.postimg.org/ok59iyp7d/Capture.png)](https://postimg.org/image/bsr3cgff9/)

Even though there are some mislabelled images, they will not affect the result that much since the dataset is relatively large. 

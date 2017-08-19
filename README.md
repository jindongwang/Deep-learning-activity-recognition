# Deep Learning for Human Activity Recognition

> Deep learning is perhaps the nearest future of human activity recognition. While there are many existing non-deep method, we still want to unleash the full power of deep learning. This repo provides a demo of using deep learning to perform human activity recognition.

## Prerequisites

- Python 3.x
- Numpy
- Tensorflow

## Dataset

There are many public datasets for human activity recognition. You can refer to this survey article [Deep learning for sensor-based activity recognition: a survey](https://arxiv.org/abs/1707.03502) to find more.

In this demo, we will use UCI HAR dataset as an  example.

This dataset can be found in [here](https://archive.ics.uci.edu/ml/datasets/Smartphone+Dataset+for+Human+Activity+Recognition+%28HAR%29+in+Ambient+Assisted+Living+%28AAL%29).

Of course, this dataset needs further preprocessing before being put into the network. I've also provided a preprocessing version of the dataset as a `.npz` file so you can focus on the network. We'll talk about this later. But it is also highly recommended you download the dataset so that you can experience all the process on your own.


| #subject | #activity | Frequency |   |
| --- | --- | --- | --- |
| 30 | 6 | 50 Hz |  |


## Network structure

What is the most influential deep structure? CNN it is. So we'll use **CNN** in our demo. Besides, since most of the signals for activity recognition is time series data, **RNN** is also needed. We'll cover that in another coming demo.

### CNN structure

Convolution + pooling + convolution + pooling +   dense + dense + dense + output

That is: 2 convolutions, 2 poolings, and 3 fully connected layers. Please check the following image.



### RNN structure

Coming soon


## Related projects

- [Must-read papers about deep learning based human activity recognition](https://github.com/jindongwang/activityrecognition/blob/master/notes/deep.md)
- [guillaume-chevalier/LSTM-Human-Activity-Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)
- [aqibsaeed/Human-Activity-Recognition-using-CNN](https://github.com/aqibsaeed/Human-Activity-Recognition-using-CNN)


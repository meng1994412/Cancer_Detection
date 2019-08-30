# Cancer Detection
## Objectives
**This is ongoing project**
Implemented Mask R-CNN model to automatically segment the boundary of potential cancerous regions (specifically Melanoma) from dermoscopic images.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.1.0
* [keras](https://keras.io/) 2.2.4
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
* [Mask-RCNN](https://github.com/matterport/Mask_RCNN)
* [imgaug](https://github.com/aleju/imgaug) 0.2.9
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/) 1.16.2

## Approaches
The skin lesion boundary segmentation dataset is from [International Skin Imaging Collaboration (ISIC) 2018 Lesions dataset](https://challenge2018.isic-archive.com/), which consists of 2594 images in JPEG format and 2594 corresponding ground-truth masks in PNG format.

The `lesions.py` contains several subclasses of the Mask R-CNN classes. We define a subclass of the `Config` class inside `mrcnn` to override and/or define any configurations we might need, which is called `LesionBoundaryConfig` ([check here](https://github.com/meng1994412/Cancer_Detection/blob/master/lesions.py#L43-L57)). The `LesionBoundaryConfig` class stores all relevant configurations when training Mask R-CNN model on the skin lesion dataset.

Just as we have a training configuration, we also have an prediction/inference configuration as well, which is called `LesionBoundaryInferenceConfig` ([check here](https://github.com/meng1994412/Cancer_Detection/blob/master/lesions.py#L60-L66)). The `LesionBoundaryInferenceConfig` class is not subclass of the Mask R-CNN `Config` class but is rather a subclass of `LesionBoundaryConfig` class used for training.

The `LesionBoundaryDataset` class ([check here](https://github.com/meng1994412/Cancer_Detection/blob/master/lesions.py#L69-L143)) is responsible for managing the lesion dataset, including loading both images and their corresponding masks from disk.

### Train & evaluate the Mask R-CNN model
The `lesions.py` has three modes, including training, predicting, and investigating. The investigating mode is used to check the training set to make sure the data is properly stored. It is a wise choice to start with a pre-trained model and then fine-tune it. Thus, Mask R-CNN model with ResNet backbone pre-trained on the COCO is used, which is named `mask_rcnn_coco.h5` ([download site](https://github.com/matterport/Mask_RCNN/releases)). We first freeze all the layers before head, and train the head for 25 epochs. After the head have start to learn patterns, we could pause the training and unfreeze all the layers before, and continue the training but with smaller learning rate.

The following command can start to train the Mask R-CNN model.
```
python lesions.py --mode train
```

However, it is recommended to investigate and debug dataset, images and masks, before starting training, in order to ensure the training process can work properly.

The following command can provide some insight into dataset before training the Mask R-CNN model.
```
python lesions.py --investigate
```

## Next step
Since I just finished first-time training to get a sense of training Mask R-CNN model, there are many modifications to be done in the future, including increase the accuracy while reducing the possible overfitting situation. 

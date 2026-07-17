import torchvision
import torch
import numpy as np
import pytest

def test_StandardNet():
    '''
    Verify that we can instantiate a StandardNet.
    '''
    import gallearn

    data_fname = gallearn.config.config['gallearn_paths']['dataset']
    data_dict = gallearn.preprocessing.load_data(data_fname)
    X = data_dict['X']
    rs = data_dict['Re']

    resnet18 = torchvision.models.resnet18()
    classifier = gallearn.cnn.StandardNet(
        lr=1.e-4,
        momentum=0.5,
        backbone=resnet18,
        dataset='test_data',
        in_channels=X.shape[1]
    )

    X_scaled = classifier.scaling_function(X)

    classifier.init_optimizer()
    y_pred = classifier(X_scaled, rs)

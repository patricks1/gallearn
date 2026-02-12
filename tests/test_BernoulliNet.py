import torchvision
import torch
import numpy as np
import pytest

@pytest.mark.filterwarnings(
        "ignore:The behavior of DataFrame concatenation with empty or all-NA"
        " entries is deprecated:FutureWarning"
    )
def test_BernoulliNet():
    '''
    Verify that we can instantiate a BernoulliNet.
    '''
    from gallearn import cnn, preprocessing

    data_fname = (
        'gallearn_data_256x256_3proj_wsat_wvmap_avg_sfr_tgt_nchw.h5'
    )
    data_dict = preprocessing.load_data(data_fname)
    X = data_dict['X']
    rs = cnn.get_radii(data_dict)

    resnet18 = torchvision.models.resnet18()
    classifier = cnn.BernoulliNet(
        lr=1.e-4,
        momentum=0.5,
        backbone=resnet18,
        dataset='test_data',
        in_channels=X.shape[1]
    )

    X_scaled = classifier.scaling_function(X)

    classifier.init_optimizer()
    y_pred = classifier(X_scaled, rs)

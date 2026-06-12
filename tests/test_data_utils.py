import pytest

@pytest.mark.filterwarnings(
        "ignore:The behavior of DataFrame concatenation with empty or all-NA"
        " entries is deprecated:FutureWarning"
    )
def test_get_radii():
    from gallearn import cnn, preprocessing

    data_fname = 'gallearn_data_256x256_3proj_wsat_wvmap_avg_sfr_tgt_nchw.h5'
    data_dict = preprocessing.load_data(data_fname)
    X = data_dict['X']
    rs = cnn.get_radii(data_dict)

    X_len = X.shape[0]
    rs_len = rs.shape[0]
    assert X_len == rs_len, (
        'The feature matrix in {0} has {1:d} samples while `cnn.get_radii`'
        ' produces {2:d} samples from {0}'.format(data_fname, X_len, rs_len)
    )

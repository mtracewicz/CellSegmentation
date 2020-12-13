import pytest
from neural_network.unet.predict import make_predictions_for_images_in_directory,make_prediction_for_image,make_prediction,save_prediction

def test_directory_on_make_prediction_for_image(tmpdir):
    with pytest.raises(IsADirectoryError) as e_info:
        make_prediction_for_image('', tmpdir)

def test_file_on_make_predictions_for_images_in_directory(tmpdir):
    with pytest.raises(NotADirectoryError) as e_info:
        f = tmpdir.join('test.txt')
        make_predictions_for_images_in_directory('', f)

def test_directory_on_make_prediction(tmpdir):
    with pytest.raises(IsADirectoryError) as e_info:
        make_prediction('',tmpdir)

def test_not_ndarray_on_save_prediction():
    with pytest.raises(TypeError) as e_info:
        save_prediction('Invalid type of argument')
import tensorflow as tf
from hyperparameters import batch_size
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Provide checkpoint name and type basic/conv")
        exit()

    # Restoring model
    model = tf.keras.models.load_model(
        f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}')
    # Loading data
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Data normalization from [0,255] to [0.0,1.0] and proper dimensions
    if(sys.argv[2] == "conv"):
        x_test_normalized = (x_test/255.0).reshape(x_test.shape[0], 28, 28, 1)
    else:
        x_test_normalized = x_test/255.0

    model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

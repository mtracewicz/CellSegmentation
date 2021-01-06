import tensorflow as tf
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: evaluate_trained_model model_name input_directory output_directory")
        exit()

    # Restoring model
    model = tf.keras.models.load_model(
        os.path.join('trained_models', sys.argv[1]))
    # Loading data

    model.evaluate(x_test, y_test)

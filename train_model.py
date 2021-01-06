import os
import sys

import tensorflow as tf

from neural_network.unet.image_loader import load_images

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python unet.py inputs_directory truths_directory checkpoint_name")
        exit()

    model_file = sys.argv[4] if len(
        sys.argv) >= 5 else 'neural_network.unet.model'
    metric_and_loss_file = sys.argv[5] if len(
        sys.argv) >= 6 else 'neural_network.unet.model'
    hyperparameters_file = sys.argv[6] if len(
        sys.argv) >= 7 else 'neural_network.unet.hyperparameters'

    exec(
        f'from {hyperparameters_file} import BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT')
    exec(f'from {model_file} import get_model')
    exec(f'from {metric_and_loss_file} import custom_metric, custom_loss')

print('Loading data')
# Loading data
x_train = load_images(sys.argv[1])
y_train = load_images(sys.argv[2])

split = int(x_train.shape[0] * (1-VALIDATION_SPLIT))

print('Seting up environment')

# Seting tf/keras options
tf.keras.backend.set_floatx('float64')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Creating model')
# Establish the model's topography
model = get_model((200, 200, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE), loss=custom_loss, metrics=[custom_metric])

print('Model info')
print(model.summary())

print('Traing model')
model.fit(x_train[:split], y_train[:split],
          batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True)

print('Model evaluation')
model.evaluate(x_train[split:], y_train[split:])

# Save model
model.save(os.path.join('trained_models', sys.argv[3]))
print('Finished!')

from keras import backend as K
from keras.losses import binary_crossentropy

from sciner.metrics import recall


def binary_crossentropy_with_sensitivity(y_true, y_pred):
    sens = recall(y_true, y_pred)
    bce = binary_crossentropy(y_true, y_pred)
    return bce + K.clip(1 - sens, K.epsilon(), 1 - K.epsilon())


if __name__ == "__main__":
    raise RuntimeError

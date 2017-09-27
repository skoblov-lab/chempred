from keras import backend as K


def specificity(y_true, y_pred):
    labels_true = K.argmax(y_true, axis=-1)
    labels_pred = K.argmax(y_pred, axis=-1)
    negative_true = K.cast(K.equal(labels_true, 0), dtype=K.floatx())
    negative_pred = K.cast(K.equal(labels_pred, 0), dtype=K.floatx())
    positive_pred = K.cast(K.equal(labels_pred, 1), dtype=K.floatx())
    true_negatives = K.sum(negative_true * negative_pred) + K.epsilon()
    false_positives = K.sum(positive_pred * negative_true) + K.epsilon()
    return true_negatives / (true_negatives + false_positives)


def sensitivity(y_true, y_pred):
    labels_true = K.argmax(y_true, axis=-1)
    labels_pred = K.argmax(y_pred, axis=-1)
    positive_true = K.cast(K.equal(labels_true, 1), dtype=K.floatx())
    positive_pred = K.cast(K.equal(labels_pred, 1), dtype=K.floatx())
    negative_pred = K.cast(K.equal(labels_pred, 0), dtype=K.floatx())
    true_positives = K.sum(positive_true * positive_pred) + K.epsilon()
    false_negatives = K.sum(positive_true * negative_pred) + K.epsilon()
    return true_positives / (true_positives + false_negatives)


def precision(y_true, y_pred):
    """
    Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """
    Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta):
    """
    Calculates the F score, the weighted harmonic mean of precision and recall.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """
    Calculates the f-measure, the harmonic mean of precision and recall.
    """
    return fbeta_score(y_true, y_pred, beta=1)

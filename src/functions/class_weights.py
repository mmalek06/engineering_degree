import numpy as np
import tensorflow as tf


def calculate_class_weight(dataset: tf.data.Dataset, alpha: float) -> dict[int, float]:
    """
    :param dataset:
    :param alpha: raw_weights can be far off from each other if the disbalance is very big, and this can
                  lead the model astray because it will focus too much on the underrpresented classes.
                  Better if it focuses on them more, but still gives some attention to the other classes.
                  This parameter should be used to balance the weights. Using 0 will make the weights
                  equal to raw_weights, and using 1 will make weights contain averages.
    :return:
    """
    class_sums = None

    for _, labels in dataset:
        batch_sum = np.sum(labels.numpy(), axis=0)
        if class_sums is None:
            class_sums = batch_sum
        else:
            class_sums += batch_sum

    total_samples = sum(class_sums)
    n_classes = len(class_sums)
    raw_weights = {i: (1 / class_sum) * total_samples / n_classes for i, class_sum in enumerate(class_sums)}
    average_weight = sum(raw_weights.values()) / n_classes
    weights = {i: alpha * average_weight + (1 - alpha) * raw_weight for i, raw_weight in raw_weights.items()}

    return weights

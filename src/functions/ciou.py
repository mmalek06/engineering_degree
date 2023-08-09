import tensorflow as tf


def ciou_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Parameters:
    y_true -- ground truth bounding box, tensor of shape (?, 4), [xmin, ymin, xmax, ymax]
    y_pred -- predicted bounding box, tensor of shape (?, 4)

    Returns:
    ciou_loss -- scalar loss, tensor of shape ()
    """
    epsilon = 1e-3
    true_y1, true_x1, true_y2, true_x2 = tf.split(y_true, 4, axis = -1)
    pred_y1, pred_x1, pred_y2, pred_x2 = tf.split(y_pred, 4, axis = -1)
    # intersection calculation:
    # take leftmost x coord and rightmost x coord, subtract to get the width and limit with
    # 0 to avoid negative values, do the same with ys
    intersect_w = tf.maximum(0.0, tf.minimum(true_x2, pred_x2) - tf.maximum(true_x1, pred_x1))
    intersect_h = tf.maximum(0.0, tf.minimum(true_y2, pred_y2) - tf.maximum(true_y1, pred_y1))
    # calculate area
    intersection = intersect_w * intersect_h
    # calculate areas of the predicted and actual bounding box
    # then calculate the union
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    union = true_area + pred_area - intersection
    # IoU calculation
    iou = intersection / (union + epsilon)  # Adding epsilon to avoid division by zero
    # distance between the box centers
    true_center_x = (true_x1 + true_x2) / 2
    true_center_y = (true_y1 + true_y2) / 2
    pred_center_x = (pred_x1 + pred_x2) / 2
    pred_center_y = (pred_y1 + pred_y2) / 2
    # from pythagorean theorem - calculate euclidean distance a^2 + b^2 = c^2
    # here the distance is still in the square units, because it doesn't matter in this
    # context since if z1 < z2 then sqrt(z1) < sqrt(z2)
    # we can save some (very little) computation time by omitting the root calculation
    center_distance = tf.square(true_center_x - pred_center_x) + tf.square(true_center_y - pred_center_y)
    # enclosing box
    enclose_x1 = tf.minimum(true_x1, pred_x1)
    enclose_y1 = tf.minimum(true_y1, pred_y1)
    enclose_x2 = tf.maximum(true_x2, pred_x2)
    enclose_y2 = tf.maximum(true_y2, pred_y2)
    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    # CIoU term
    ciou_term = (1 - iou) + center_distance / (tf.square(enclose_w) + tf.square(enclose_h) + epsilon)

    return tf.reduce_mean(ciou_term)


def ciou_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    ciou_term = ciou_loss(y_true, y_pred)

    return 1.0 - tf.reduce_mean(ciou_term) # higher value is better

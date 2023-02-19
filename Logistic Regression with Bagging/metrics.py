"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""


def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp, tn, fp, fn = calculate_vars(y_true, y_pred)
    return (tp + tn)/ (tp + tn + fp + fn)
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp, _, fp, __ = calculate_vars(y_true, y_pred)
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp, _, __, fn = calculate_vars(y_true, y_pred)
    return tp/ (tp + fn)


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp, _, fp, fn = calculate_vars(y_true, y_pred)
    return 2*tp/ (2*tp + fp + fn)


def calculate_vars(y_true, y_pred):

    tp, tn, fp, fn = 0, 0, 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp +=1
        if true == 0 and pred == 0:
            tn +=1
        if true == 0 and pred == 1:
            fp +=1
        if true == 1 and pred == 0:
            fn +=1
    return tp, tn, fp, fn
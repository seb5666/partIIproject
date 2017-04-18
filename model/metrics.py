from keras import backend as K
from sklearn.metrics import precision_score

def  dice(y_true, y_pred):
    # Symbolically compute the intersection
    y_int = y_true*y_pred
    # Technically this is the negative of the Sorensen-Dice index. This is done for minimization purposes
    return -(2*K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))

def per_class_precision(y_true, y_pred):
    print(K.shape(y_true))
    print(K.shape(y_pred))
    y_true = K.eval(y_true)
    y_pred = K.eval(y_pred)
    print(y_true.shape)
    num_classes = 6
    names = ["class " + str(i) for i in range(num_classes)]
    print(names)
    precisions = [K.variable(score) for score in precision_score(y_true, y_pred, average=None)]
    print(precisions)
    return dict(zip(names, precisions))


from sklearn.metrics import accuracy_score

def accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)
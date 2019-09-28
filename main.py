"""
Reza Marzban
"""


from utils import load_data, make_features_binary, make_data_binary, create_subset
from utils import NaiveBayes, NaiveBayesGaussian, NearestNeighbors



def problem1():
    nb = NaiveBayes()
    x_train, y_train, x_test, y_test = load_data()
    x_train, x_test = make_features_binary(x_train, x_test)
    nb.fit(x_train, y_train)
    accuracy = nb.predict_and_evaluate(x_test, y_test)
    print(f"Naive Bayes accuracy (with Dirichlet prior): {accuracy}")


def problem2():
    nbg = NaiveBayesGaussian()
    train_x, train_y, _, _ = load_data()
    x_train, y_train, x_test, y_test = make_data_binary(train_x, train_y)
    nbg.fit(x_train, y_train)
    y_hat = nbg.predict_and_evaluate(x_test, y_test)
    print()


def problem3():
    nn = NearestNeighbors()
    x_train, y_train, x_test, y_test = load_data()
    x_train, y_train, x_test, y_test = create_subset(x_train, y_train, x_test, y_test)
    validation_acc = nn.fit(x_train, y_train)
    print(f"Best K = {nn.best_k}")
    print(f"Validation Accuracy = {validation_acc}")
    test_acc, predictions = nn.predict_and_evaluate(x_test, y_test)
    print(f"Test Accuracy = {test_acc}")
    nn.visualise_classification(predictions)
    print()


if __name__ == "__main__":
    problem1()
    problem2()
    problem3()

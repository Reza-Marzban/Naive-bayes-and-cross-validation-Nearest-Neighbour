"""
Reza Marzban
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def load_data():
    # load the mnist data, and transform into right shape numpy array
    training_images_file = open('train-images.idx3-ubyte', 'rb')
    training_images = training_images_file.read()
    training_images_file.close()
    training_images = bytearray(training_images)
    training_images = training_images[16:]
    training_images = np.array(training_images)
    training_images = training_images.reshape(-1, 784)

    training_labels_file = open('train-labels.idx1-ubyte', 'rb')
    training_labels = training_labels_file.read()
    training_labels_file.close()
    training_labels = bytearray(training_labels)
    training_labels = training_labels[8:]
    training_labels = np.array(training_labels)
    training_labels = training_labels[:, np.newaxis]

    testing_images_file = open('t10k-images.idx3-ubyte', 'rb')
    testing_images = testing_images_file.read()
    testing_images_file.close()
    testing_images = bytearray(testing_images)
    testing_images = testing_images[16:]
    testing_images = np.array(testing_images)
    testing_images = testing_images.reshape(-1, 784)

    testing_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    testing_labels = testing_labels_file.read()
    testing_labels_file.close()
    testing_labels = bytearray(testing_labels)
    testing_labels = testing_labels[8:]
    testing_labels = np.array(testing_labels)
    testing_labels = testing_labels[:, np.newaxis]

    return training_images, training_labels, testing_images, testing_labels


def visualize_datapoint(data, title):
    data = data.reshape(28, 28)
    plt.title(title)
    plt.imshow(data, cmap='gray')
    plt.show()


def make_features_binary(trainx, testx):
    threshhold = 110
    trainx = (trainx >= threshhold) * 1
    testx = (testx >= threshhold) * 1
    return trainx, testx


def make_data_binary(train_x, train_y):
    five_index = np.where(train_y == 5)[0]
    rest_index = np.where(train_y != 5)[0]
    five_index = np.random.choice(five_index, size=1000)
    rest_index = np.random.choice(rest_index, size=1000)
    x = np.concatenate((train_x[five_index], train_x[rest_index]))
    y = np.concatenate((train_y[five_index], train_y[rest_index]))
    y[y != 5] = 0
    y[y == 5] = 1
    mask = np.random.rand(len(x)) < 0.90
    training_images = x[mask]
    training_labels = y[mask]
    mask = np.logical_not(mask)
    testing_images = x[mask]
    testing_labels = y[mask]
    return training_images, training_labels, testing_images, testing_labels


def create_subset(train_x, train_y, test_x, test_y):
    train_size_each_class = 200
    one_index = np.where(train_y == 1)[0]
    one_index = np.random.choice(one_index, size=train_size_each_class)
    two_index = np.where(train_y == 2)[0]
    two_index = np.random.choice(two_index, size=train_size_each_class)
    seven_index = np.where(train_y == 7)[0]
    seven_index = np.random.choice(seven_index, size=train_size_each_class)
    xtrain = np.concatenate((train_x[one_index], train_x[two_index], train_x[seven_index]))
    ytrain = np.concatenate((train_y[one_index], train_y[two_index], train_y[seven_index]))

    test_size_each_class = 50
    one_index = np.where(test_y == 1)[0]
    one_index = np.random.choice(one_index, size=test_size_each_class)
    two_index = np.where(test_y == 2)[0]
    two_index = np.random.choice(two_index, size=test_size_each_class)
    seven_index = np.where(test_y == 7)[0]
    seven_index = np.random.choice(seven_index, size=test_size_each_class)
    xtest = np.concatenate((test_x[one_index], test_x[two_index], test_x[seven_index]))
    ytest = np.concatenate((test_y[one_index], test_y[two_index], test_y[seven_index]))
    return xtrain, ytrain, xtest, ytest


class NaiveBayes:
    """
    Naive Bayes class on binary features.
    """
    _mu = None
    _theta = None

    def fit(self, x, y):
        """
        :param x: training set features
        :param y: training set labels
        """
        N = len(x)
        k, N_k = np.unique(y, return_counts=True)
        d = len(x[0])
        mu = np.append(k.reshape(len(k), 1), N_k.reshape(len(N_k), 1)/N, 1)
        theta = np.ones((len(k), d))
        for image in range(N):
            mask = x[image] > 0
            theta[y[image], mask] += 1
        theta = theta / (N_k[:, None]+d)
        self._mu = mu
        self._theta = theta

    def classify(self, image):
        """
        :param image: input image
        :return: label: predicted label
        """
        label = None
        max_prob = -10e7
        for k in range(len(self._mu)):
            prob_k = math.log(self._mu[k][1])
            N_K = float(np.sum(self._theta[k]))
            true_pixels = np.where(image == 1)
            false_pixels = np.where(image == 0)
            p = self._theta[k][true_pixels]
            p1 = 1-self._theta[k][false_pixels]
            log_likelihood = np.log(p).sum() + np.log(p1).sum()
            prob = prob_k + log_likelihood
            if prob > max_prob:
                max_prob = prob
                label = k
        return label

    def predict_and_evaluate(self, x, y):
        """
        :param x: testing set features
        :param y: testing set labels
        :return: accuracy: prediction accuracy
        """
        correct_prediction_counter = 0
        if self._mu is None or self._theta is None:
            print("Usage Error: Please use NaiveBayes.fit(x,y) first.")
            return -1
        for i in range(len(x)):
            image = x[i]
            true_y = y[i]
            y_hat = self.classify(image)
            if true_y == y_hat:
                correct_prediction_counter += 1
        accuracy = round(correct_prediction_counter/len(x), 4)
        return accuracy


class NaiveBayesGaussian:
    """
        Gaussian Naive Bayes class.
    """
    _v = None
    _mu = None
    _prior = None

    def fit(self, x, y):
        """
        :param x: training set features
        :param y: training set labels
        """
        N = len(x)
        k, N_k = np.unique(y, return_counts=True)
        self._prior = np.append(k.reshape(len(k), 1), N_k.reshape(len(N_k), 1) / N, 1)
        training_set_with_five = x[(y == 1).squeeze()]
        training_set_without_five = x[(y == 0).squeeze()]
        v0 = np.var(training_set_without_five)
        v1 = np.var(training_set_with_five)
        mu = np.stack((np.mean(training_set_without_five, axis=0), np.mean(training_set_with_five, axis=0)))
        self._v = (v0, v1)
        self._mu = mu

    def _probability_density_function(self, image):
        """
        :param image: input image
        :return: Probability density functtion for both label 0 and 1.
        """
        v0, v1 = self._v
        pdf0 = 1/math.sqrt(2*math.pi*v0)
        e0 = np.exp((-1 * np.power((image - self._mu[0]), 2)) / (2 * v0))
        pdf0 *= e0
        pdf1 = 1/math.sqrt(2*math.pi*v1)
        e1 = np.exp((-1 * np.power((image - self._mu[1]), 2)) / (2 * v1))
        pdf1 *= e1
        return pdf0, pdf1

    def classify(self, image):
        """
        :param image: input image
        :return: label: predicted label
        """
        log_prior_0 = math.log(self._prior[0][1])
        log_prior_1 = math.log(self._prior[1][1])
        pdf0, pdf1 = self._probability_density_function(image)
        log_pdf0 = np.log(pdf0).sum()
        log_pdf1 = np.log(pdf1).sum()
        p0 = log_prior_0 + log_pdf0
        p1 = log_prior_1 + log_pdf1
        score = p1-p0
        if p1 > p0:
            label = 1
        else:
            label = 0
        return label, score

    @staticmethod
    def _check_threshold(y, scores):
        y_true = (y == 1)
        sorted_score_indices = np.argsort(scores)[::-1]
        y_score = scores[sorted_score_indices]
        y_true = y_true[sorted_score_indices]
        unique_value_indices = np.where(np.diff(y_score))[0]
        threshold_indices = np.r_[unique_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_indices]
        fps = 1 + threshold_indices - tps
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        fpr = fps / fps[-1]
        tpr = tps / tps[-1]
        return fpr, tpr

    def predict_and_evaluate(self, x, y):
        """
        :param x: testing set features
        :param y: testing set labels
        """
        if self._mu is None or self._v is None or self._prior is None:
            print("Usage Error: Please use NaiveBayesGaussian.fit(x,y) first.")
            return -1
        y_hat = []
        scores = []
        TP, TN, FN, FP = (0, 0, 0, 0)
        for i in range(len(x)):
            image = x[i]
            true_y = y[i]
            prediction, score = self.classify(image)
            if true_y == 1 and prediction == 1:
                TP += 1
            elif true_y == 1 and prediction == 0:
                FN += 1
            elif true_y == 0 and prediction == 1:
                FP += 1
            elif true_y == 0 and prediction == 0:
                TN += 1
            scores.append(score)
            y_hat.append(prediction)
        y_hat = np.array(y_hat)
        scores = np.array(scores)
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        accuracy = (TP+TN)/(len(x))

        FPRs, TPRs = self._check_threshold(y, scores)
        auc = round(np.trapz(TPRs, FPRs), 4)

        plt.plot(FPRs, TPRs, linewidth=3.5)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title(f"ROC Curve -- AUC= {auc}", fontsize=16)
        plt.ylabel('True Positive Rate (TPR)', fontsize=14)
        plt.xlabel('False Positive Rate (FPR)', fontsize=14)
        plt.grid()
        plt.show()
        return y_hat


class NearestNeighbors:
    """
    Nearest Neighbors class (Brute Force).
    """
    # k_list = [1, 3, 5, 7, 9]
    k_list = [3]
    best_k = None
    Training_x = None
    Training_y = None
    Testing_x = None
    Testing_y = None

    @staticmethod
    def _euclidean_distance(n_points, x):
        n_points = n_points.astype('float64')
        x = x.astype('float64')
        eucl_dist = (n_points - x)
        eucl_dist = eucl_dist**2
        eucl_dist = np.sum(eucl_dist, axis=1)
        eucl_dist = np.sqrt(eucl_dist)
        return eucl_dist

    @staticmethod
    def _shuffle_data(x, y):
        assert len(x) == len(y)
        p = np.random.permutation(len(x))
        return x[p], y[p]

    def _classify(self, train_x, train_y, validation_x, validation_y, k):
        predictions = []
        for i in range(len(validation_x)):
            x = validation_x[i]
            label = validation_y[i]
            distances = self._euclidean_distance(train_x, x)
            nearest_indices = distances.argsort()[:k]
            nearest_indices = np.array(nearest_indices)
            nearest_labels = train_y[nearest_indices]
            _, idx, counts = np.unique(nearest_labels, return_index=True, return_counts=True)
            index = idx[np.argmax(counts)]
            mode = nearest_labels[index]
            prediction = mode
            predictions.append(prediction)
        predictions = np.array(predictions)
        accuracy = round((predictions == validation_y).sum() / len(validation_y), 4)
        return accuracy, predictions

    def _cross_validate_nn(self, x, y, folds_count):
        folds_x = np.array_split(x, folds_count)
        folds_y = np.array_split(y, folds_count)
        accuracy_sum = np.zeros((len(self.k_list)))
        for i in range(folds_count):
            test_set_x = np.array(folds_x[i])
            test_set_y = np.array(folds_y[i])
            training_set_x = None
            training_set_y = None
            accuracy_list = []
            for j in range(folds_count):
                if i == j:
                    continue
                if training_set_x is None:
                    training_set_x = np.vstack((folds_x[j]))
                    training_set_y = np.vstack((folds_y[j]))
                else:
                    training_set_x = np.vstack((training_set_x, folds_x[j]))
                    training_set_y = np.vstack((training_set_y, folds_y[j]))
            accuracy_list = []
            for k in self.k_list:
                accuracy, _ = self._classify(training_set_x, training_set_y, test_set_x, test_set_y, k)
                accuracy_list.append(accuracy)
            accuracy_list = np.array(accuracy_list)
            accuracy_sum += accuracy_list
        accuracy_ave = accuracy_sum / folds_count
        best_k_idx = np.argmax((accuracy_ave))
        self.best_k = self.k_list[best_k_idx]
        return accuracy_ave[best_k_idx]

    def fit(self, x, y):
        """
        :param x: training set features
        :param y: training set labels
        """
        x, y = self._shuffle_data(x, y)
        self.Training_x = x
        self.Training_y = y
        accuracy = round(self._cross_validate_nn(x, y, 5),4)
        return accuracy

    def predict_and_evaluate(self, x, y):
        """
        :param x: testing set features
        :param y: testing set labels
        """
        if self.best_k is None or self.Training_x is None or self.Training_y is None:
            print("Usage Error: Please use NearestNeighbors.fit(x,y) first.")
            return -1
        x, y = self._shuffle_data(x, y)
        self.Testing_x = x
        self.Testing_y = y
        k = self.best_k
        accuracy, predictions = self._classify(self.Training_x, self.Training_y, x, y, k)
        return accuracy, predictions

    def visualise_classification(self, y_hat):
        if self.Testing_x is None or self.Testing_y is None:
            print("Usage Error: Please use NearestNeighbors.predict_and_evaluate(x,y) first.")
            return -1
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.3)
        x = self.Testing_x
        true_y = self.Testing_y
        try:
            i1t = np.where(np.logical_and(true_y == 1, y_hat == 1))[0][0]
            i1f = np.where(np.logical_and(true_y != 1, y_hat == 1))[0][0]
            i2t = np.where(np.logical_and(true_y == 2, y_hat == 2))[0][0]
            i2f = np.where(np.logical_and(true_y != 2, y_hat == 2))[0][0]
            i7t = np.where(np.logical_and(true_y == 7, y_hat == 7))[0][0]
            i7f = np.where(np.logical_and(true_y != 7, y_hat == 7))[0][0]
        except:
            print()
            print("Not enough misclassified data to generate visualization! Please Run Problem3 again.")
            return
        title_size = 7
        fig.add_subplot(2, 3, 1)
        data = x[i1t]
        title = "Correctly Classified 1"
        data = data.reshape(28, 28)
        plt.title(title, fontsize=title_size)
        plt.imshow(data, cmap='gray')
        fig.add_subplot(2, 3, 4)
        data = x[i1f]
        title = "Incorrectly Classified 1"
        data = data.reshape(28, 28)
        plt.title(title, fontsize=title_size)
        plt.imshow(data, cmap='gray')

        fig.add_subplot(2, 3, 2)
        data = x[i2t]
        title = "Correctly Classified 2"
        data = data.reshape(28, 28)
        plt.title(title, fontsize=title_size)
        plt.imshow(data, cmap='gray')
        fig.add_subplot(2, 3, 5)
        data = x[i2f]
        title = "Incorrectly Classified 2"
        data = data.reshape(28, 28)
        plt.title(title, fontsize=title_size)
        plt.imshow(data, cmap='gray')

        fig.add_subplot(2, 3, 3)
        data = x[i7t]
        title = "Correctly Classified 7"
        data = data.reshape(28, 28)
        plt.title(title, fontsize=title_size)
        plt.imshow(data, cmap='gray')
        fig.add_subplot(2, 3, 6)
        data = x[i7f]
        title = "Incorrectly Classified 7"
        data = data.reshape(28, 28)
        plt.title(title, fontsize=title_size)
        plt.imshow(data, cmap='gray')
        plt.show()


if __name__ == "__main__":
    print("utils.py: a helper file for main.py.\n")

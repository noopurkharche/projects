import math
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Reference: https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
def main(argv):
    print("main")

    mode = argv[1]
    # get the learning rate
    learningRate = float(argv[2])

    # generating first training set of 1000
    mean = [1, 0]

    n = 1000

    # covariance
    cov = [[1, 0.75], [0.75, 1]]

    x1 = np.random.multivariate_normal(mean, cov, n)

    # generating second training set of 1000
    mean = [0, 1.5]

    x2 = np.random.multivariate_normal(mean, cov, n)

    trainingData = np.concatenate((x1,x2))

    # add labels first 1000 as 0 and next 1000 as 1
    label = np.array([0] * n + [1] * n)

    # add ones at the start
    ones = np.ones((2*n,1))

    data = np.concatenate((ones,trainingData), axis=1)

    #learningRate = 1

    if mode == 'batch':
        leranedWeights = batch_training(data, learningRate, label)
        accuracy = calculateAccuracy(data, label, leranedWeights)
        print("Accuracy : " + str((accuracy/float(len(data)))*100))
    else:
        leranedWeights = online_training(data, learningRate, label)
        accuracy = calculateAccuracy(data, label, leranedWeights)
        print("Accuracy : " + str((accuracy/float(len(data)))*100))


# method to calculate the accuracy
# para[in] : data - testing data
# para[in] : label - labels list
# para[in] : learnedWeights - Edge weights learned from training
def calculateAccuracy(data, label, leranedWeights):

    correctCount = 0
    count = 0
    for j in range(len(data)):
        z = data[j].dot(leranedWeights)
        predicted = 1 / (1 + np.exp(-z))

        if predicted >= 0.5:
            predictedLabel = 1
        else:
            predictedLabel = 0

        if predictedLabel == label[j]:
            correctCount = correctCount + 1

    return correctCount


# method for batch Training
# para[in] : data - training data
# para[in]: learningRate - learning rate
# para[in]: label - list of labels 0 or 1
# para[out]: returns the learned weights
def batch_training(data, learningRate, label):

    weights = [1,1,1]
    oldError = 0

    tpr = []
    fpr = []

    for i in range(10000):

        error = 0

        # calculate sigmoid value logisic regression
        z = data.dot(weights)
        predicted = 1/(1 + np.exp(-z))

        # number of True Positives
        numberOfTP = 0
        # number of True Negatives
        numberOfTN = 0
        # number of False Positives
        numberOfFP = 0
        # number of False Negatives
        numberOfFN = 0

        for j in range(len(data)):

            # calculate the cross entropy objective function
            if label[j] == 1:
                error = error - np.log(predicted[j])
            else:
                error = error - np.log(1-predicted[j])

            # for calculation of 2-class confusion matrix values
            if predicted[j] >= 0.5:
                predictedlabel = 1
            else:
                predictedlabel = 0

            actualLabel = label[j]

            # calculate the 2-class confusion matrix values
            if(predictedlabel == 0 and actualLabel == 0):
                numberOfTP = numberOfTP + 1

            if (predictedlabel == 0 and actualLabel == 1):
                numberOfFP = numberOfFP + 1

            if (predictedlabel == 1 and actualLabel == 0):
                numberOfFN = numberOfFN + 1

            if (predictedlabel == 1 and actualLabel == 1):
                numberOfTN = numberOfTN + 1

        # calculate the gradient descent numerator value using (o - y) * x
        numerator = np.dot(data.T, predicted - label)
        gradient = numerator/len(data)

        # calculate the new weights
        gradient = gradient * learningRate
        weights = weights - gradient

        # Calculate the True Positive Rate
        truePositiveRate = numberOfTP/(numberOfTP+numberOfFN+0.0)

        # Calculate the False Positive Rate
        falsePositiveRate = numberOfFP/(numberOfFP+numberOfTN+0.0)

        # maintain list of points for TP rate and FP rate
        tpr.append(truePositiveRate)
        fpr.append(falsePositiveRate)

        # stopping condition
        if oldError == error:
            break
        else:
            oldError = error
        # stopping condition
        if gradient.mean() < 0.00001:
            break

    tpr.sort()
    fpr.sort()

    print("Total iterations = " + str(i+1))
    print("Edge Weights Learned = " + str(weights))
    area = area_calculations(fpr, tpr)
    print("Area Under the Curve: " + str(area))
    plot_ROC(fpr, tpr, area)
    return weights

# method for online Training
# para[in] : data - training data
# para[in]: learningRate - learning rate
# para[in]: label - list of labels 0 or 1
# para[out]: returns the learned weights
def online_training(data, learningRate, label):

    print("Online Training: ")
    weights = [1,1,1]
    oldError = 0
    for i in range(10000):
        error = 0
        for j in range(len(data)):

            # calculate sigmoid value logisic regression
            z = data[j].dot(weights)
            predicted = 1/(1 + np.exp(-z))

            # calculate the cross entropy objective function
            if label[j] == 1:
                error = error - np.log(predicted)
            else:
                error = error - np.log(1 - predicted)

            # calculate the gradient descent numerator value using (o - y) * x
            numerator = (np.dot(data[j].T, predicted - label[j]))
            gradient = numerator/len(data)

            # calculate the new weights
            gradient = gradient * learningRate
            weights = weights - gradient

        # stopping condition
        if oldError == error:
            break

        #if i == 7000:
        #    print(gradient.mean())
        #    break
        # stopping condition

        if gradient.mean() < -0.1:
            print(gradient.mean())
            break

    print("Total iterations = " + str(i + 1))
    print("Edge Weights Learned = " + str(weights))
    return weights


# method to calculate the area under thee curve
# para[in] : x - list of FPR points
# para[in] : y - list of TPR points
# returns the area under the curve
def area_calculations(x, y):

    area = 0
    for i in xrange(1, len(x)):
        area += (x[i] - x[i - 1]) * ((y[i - 1] + y[i]) / 2.0)
    return area


# Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# method to plot the ROC
# para[in] : fpr - false positive rate
# para[in] : tpr - true positive rate
def plot_ROC(fpr, tpr, area):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='red',
             lw=lw, label='ROC curve (area =' + str(area) + ')')
    plt.plot([0, 0.7], [0, 0.7], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 0.1])
    plt.ylim([0.0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC for Batch Training')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
        main(sys.argv)

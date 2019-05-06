Implement Logistic Regression using perceptron.

Implemented a perceptron for logistic regression in python. Used sigmoid function as activation function and cross-entropy as an objective function. Performed both batch Training and Online Training using Gradient Descent. Also, plotted ROC and calculated the Area Under ROC to evaluate the performance of LR classifier.

Programing language: Python

TO run the code Just type in the following command 

python filename.py [mode] [learning rate]

There are two modes to run batch mode and online mode.
if you pass batch, then program runs for all batch training and also plot the ROC 
if you pass online, then program runs for all online training


To run batch training,

learning rate = 1

	python assignment2.py batch 1

learning rate = 0.1 

	python assignment2.py batch 0.1

learning rate = 0.01

	python assignment2.py batch 0.01

To run online training,

learning rate = 1

	python assignment2.py online 1

learning rate = 0.1

	python assignment2.py online 0.1

learning rate = 0.01

	python assignment2.py online 0.01

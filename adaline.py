import random
from sommelier import matrix_multiply
from sommelier import dot_product
from sommelier import transpose


class Adaline:
    def __init__(self):
        self.W = None
        self.learning_rate = 0
        self.cost = []
        self.performance = []

    def validate_weights(self, X, y):
        """
        This function will validate our model on a validation set gathered using the k_fold sampling method.
        It will return the percentage of correctly labeled observations from the validation set after training our model
        on a training set.

        :param X: 2Dimensional array representing our validation set shape: (# of observations, # of features)
        :param y: pandas.series object with the correct labels for our validation set
        :return: float: accuracy of our model
        """
        outputs = self.predict(X)
        num_errors = 0

        for yi, output in zip(y, outputs):
            if yi != output:
                num_errors += 1

        return 1.0 - float(num_errors) / float(len(outputs))

    def get_performance(self, X, y, epoch):
        """
        Calculate performance statistics for given epoch and add results to self.performance list
        """
        results = self.predict(X)
        num_errors = 0

        for yi, output in zip(y, results):
            if yi != output:
                num_errors += 1

        performance = (epoch, num_errors, self.W[1:], self.W[0])
        self.performance.append(performance)

    def predict(self, X):
        """
        This functions returns a list of predictions using the input matrix X.
        We label a wine as good quality if the output using our linear activation function is greater than 5

        :param X: matrix
        :return: list of booleans
        """
        results = []
        for xi in zip(X):
            x = [xi[0][i] for i in range(len(xi[0]))]
            output = self.output(x)
            results.append(output > 0.5)
        return results

    def converged(self):
        """
        This function checks if our model's weights have converged to a good set of weights.
        Weights are converged if the difference between costs of the last two epochs is less than 0.000001

        :return: boolean
        """
        if len(self.cost) > 1:
            diff = self.cost[-1] - self.cost[-2]
            diff = diff if diff > 0.0 else -diff
            if diff < 0.000001:
                return True
        return False

    def output(self, xi):
        """
        Calculate the output of the given input xi. xi is a vector representing one observation.
        We are using a linear activation function.

        :param xi: series
        :return: float
        """
        output = dot_product(xi, self.W[1:]) + self.W[0]
        return output

    def train_epoch(self, X, y, batch_learning, epoch):
        """
        This function will run one epoch of our training using the inputs X and the labels y.
        Batch_learning is a boolean that indicates if we will do batch learning or online learning.

        :param X: matrix
        :param y: series
        :param batch_learning: boolean
        :param epoch: int
        :return : void
        """
        errors = []

        # predict output for each vector in our input matrix. Calculate error and add error to errors list
        # update weights if we are doing online learning
        for xi, yi in zip(X, y):
            error = yi - self.output(xi)
            errors.append(error)
            if not batch_learning:
                self.W[0] += self.learning_rate * error
                self.W[1:] += self.learning_rate * error * xi

        # update weights if we are dong batch learning
        if batch_learning:
            self.W[0] += sum(errors) * self.learning_rate
            updates = [x * self.learning_rate for x in matrix_multiply(transpose(X), errors)]
            self.W[1:] = [w + u for w, u in zip(self.W[1:], updates)]

        # calculate total cost for this epoch and add to self.cost list
        cost = [err**2 for err in errors]
        self.cost.append(sum(cost) / 2.0)

        # get performance statistics for this epoch and add it to performance list
        self.get_performance(X, y, epoch)

    def train(self, X, y, num_epochs, learning_rate, batch_learning=True):
        """
        This function will initialize all the variables for out Adaline model.
        Then it will train the weights using the inputs X and the labels y.
        If number of epochs is 0 then we will train until our weights converge.

        :param X: Matrix representing then inputs for our model
        :param y: series representing the labels that will be used to calculate the errors on our prediction
        :param num_epochs: int
        :param learning_rate: float
        :param batch_learning: boolean, if False we will use online learning
        :return: void
        """

        # initialize Adaline instance variables, generate random numbers for weights and bias
        self.learning_rate = learning_rate
        self.W = [random.uniform(-1, 1) for i in range(X.shape[1] + 1)]
        self.cost = []
        self.performance = []

        epoch = 0
        if num_epochs < 0:
            raise ValueError("number of epochs must be non-negative")

        # train model
        while True:
            self.train_epoch(X, y, batch_learning, epoch)
            epoch += 1
            if num_epochs == 0 and self.converged():
                break
            if num_epochs != 0 and epoch >= num_epochs:
                break

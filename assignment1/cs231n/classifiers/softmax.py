from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]

    for i in range(len(X)):
        scores = X[i].dot(W)
        # For better numerical stability, subtract max of scores.
        # This doesn't affect resulting loss and dW
        scores = scores - np.max(scores) 
        exp_scores = np.exp(scores)
        exp_scores_sum = exp_scores.sum()
        loss += - np.log(exp_scores[y[i]] / exp_scores_sum)

        for j in range(num_classes):
            dW[:, j] += X[i] * (exp_scores[j] / exp_scores_sum - int(j == y[i]))

    loss /= len(X)
    loss += reg * np.sum(W * W)
    dW /= len(X)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = len(X)
    num_classes = W.shape[1]
    scores = X.dot(W)
    # For better numerical stability, subtract max of scores (for each sample)
    # This doesn't affect resulting loss and dW    
    scores -= scores.max(axis=1).reshape(-1, 1)
    exp_scores = np.exp(scores)
    exp_scores_sum = exp_scores.sum(axis=1)
    P = exp_scores / exp_scores_sum.reshape(-1, 1)
    Y = np.zeros_like(P)
    Y[np.arange(num_samples), y] = 1

    loss = -np.sum(np.log(P[np.arange(num_samples), y]))
    dW += X.transpose().dot(P - Y)

    loss /= num_samples
    loss += reg * np.sum(W * W)
    dW /= num_samples
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

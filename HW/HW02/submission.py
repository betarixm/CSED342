#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        pretty, good, bad, plot, not, scenery
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'pretty': 1, 'good': 0, 'bad': -1, 'plot': -1, 'not': -1, 'scenery': 0}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    return dict(Counter(x.split()))
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    def nll_grad(_phi, _y):
        return -_y * sigmoid(-_y * dotProduct(_phi, weights))

    for _ in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            increment(weights, -eta * nll_grad(phi, y), phi)

# END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features

def extractBigramFeatures(x):
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
    words = f"<s> {x} </s>".split()
    phi = dict(Counter(words[1:-1] + [(cur_word, next_word) for cur_word, next_word in zip(words[:-1], words[1:])]))
    # END_YOUR_ANSWER
    return phi

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu0' and 'mu1'.
    Assume the initial centers are
    ({'mu0': 1, 'mu1': -1}, {'mu0': 1, 'mu1': 2})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu0': 1, 'mu1': 0}, {'mu0': 1, 'mu1': 1.5}
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu0' and 'mu1'.
    Assume the initial centers are
    ({'mu0': -2, 'mu1': 0}, {'mu0': 3, 'mu1': -1})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu0': 0, 'mu1': 0.5}, {'mu0': 2, 'mu1': 1}
    # END_YOUR_ANSWER

############################################################
# Problem 3b: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)

    cache_point = {}
    for key, value in enumerate(examples):
        examples[key] = dict(value)
        cache_point[key] = dotProduct(examples[key], examples[key])

    assignments = {}
    centers = {i: s.copy() for i, s in enumerate(random.sample(examples, K))}
    total_cost = 0
    prev_cost = -1

    def dist(p_dot, q_dot, p, q):
        return p_dot + q_dot - 2 * dotProduct(p, q)

    for __ in range(maxIters):
        if prev_cost == total_cost:
            break

        total_cost, prev_cost = 0, total_cost

        group_list = [[] for _ in range(K)]
        counter_list = [Counter() for _ in range(K)]

        cache_center = {idx: dotProduct(center, center) for idx, center in centers.items()}

        for idx_point, point in enumerate(examples):
            dist_list = [(idx, dist(cache_point[idx_point], cache_center[idx], point, center)) for idx, center in centers.items()]
            idx_center, cost = min(dist_list, key=lambda x: x[1])

            total_cost += cost

            assignments[idx_point] = idx_center
            group_list[idx_center].append(idx_point)

            for key, value in point.items():
                counter_list[idx_center][key] += value

        for idx_center in range(K):
            centers[idx_center] = {key: value / len(group_list[idx_center]) for key, value in dict(counter_list[idx_center]).items()}

    return centers, assignments, total_cost
    # END_YOUR_ANSWER


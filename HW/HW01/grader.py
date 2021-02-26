#!/usr/bin/env python

import graderUtil, collections, random

grader = graderUtil.Grader()
submission = grader.load('submission')


try:
    import solution
    grader.addHiddenPart = grader.addBasicPart
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False

def test_correct(func_name, assertion=lambda pred: True):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(pred == answer)
    return test

def test_wrong(func_name, assertion=lambda pred: True):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(pred != answer and pred is not None)
    return test

############################################################
# Problems 1a and 1b

grader.addHiddenPart('1a-1-hidden', test_correct('problem_1a', lambda pred: (0 <= pred <= 1)), 3)

grader.addHiddenPart('1b-1-hidden', test_correct('problem_1b', lambda pred: (pred in range(1, 5))), 3)
grader.addHiddenPart('1b-2-hidden', test_wrong('problem_1b', lambda pred: (pred in range(1, 5))),
                     -1, extraCredit=True)

############################################################
# Problem 2a: getWordKey for computeMaxWordLength

def computeMaxWordLength(text, customKey):
    return max(text.split(), key=customKey)

def test():
    grader.requireIsEqual('longest', computeMaxWordLength('which is the longest word', submission.getWordKey))
    grader.requireIsEqual('sun', computeMaxWordLength('cat sun dog', submission.getWordKey))
    grader.requireIsEqual('99999', computeMaxWordLength(' '.join(str(x) for x in range(100000)), submission.getWordKey))

grader.addBasicPart('2a-0-basic', test)

def test():
    random.seed(SEED)
    chars = tuple(map(lambda x: chr(x), range(ord('a'), ord('z') + 1)))
    def get_gen():
        for _ in range(20):
            text = ' '.join(''.join(random.choice(chars) for _ in range(random.choice(range(5, 25))))
                            for _ in range(500))
            pred = computeMaxWordLength(text, submission.getWordKey)
            if solution_exist:
                answer = computeMaxWordLength(text, solution.getWordKey)
                yield grader.requireIsTrue(pred == answer)
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('2a-1-hidden', test, 3)

############################################################
# Problem 2b: manhattanDistance

grader.addBasicPart('2b-0-basic', lambda : grader.requireIsEqual(6, submission.manhattanDistance((3, 5), (1, 9))))

def test():
    random.seed(SEED)
    def get_gen():
        for _ in range(100):
            x = tuple(random.randint(0, 10) for _ in range(10))
            y = tuple(random.randint(0, 10) for _ in range(10))
            pred = submission.manhattanDistance(x, y)
            if solution_exist:
                answer = solution.manhattanDistance(x, y)
                yield grader.requireIsTrue(pred == answer)
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('2b-1-hidden', test, 2)

############################################################
# Problem 2c: countMutatedSentences

def test():
    grader.requireIsEqual(1, submission.countMutatedSentences('a a a a a'))
    grader.requireIsEqual(1, submission.countMutatedSentences('the cat'))
    grader.requireIsEqual(4, submission.countMutatedSentences('the cat and the mouse'))
grader.addBasicPart('2c-0-basic', test)

def genSentence(K, L): # K = alphabet size, L = length
    return ' '.join(str(random.randint(0, K)) for _ in range(L))
def get_test(K, L, N):
    random.seed(SEED)  # N = num of repeats
    def get_gen():
        for _ in range(N):
            sentence = genSentence(K, L)
            pred = submission.countMutatedSentences(sentence)
            if solution_exist:
                answer = solution.countMutatedSentences(sentence)
                yield grader.requireIsTrue(pred == answer)
            else:
                yield True
    return lambda: all(get_gen())

grader.addHiddenPart('2c-1-hidden', get_test(3, 20, 1), maxPoints=3, maxSeconds=5)

grader.addHiddenPart('2c-2-hidden', get_test(25, 40, 10), maxPoints=3, maxSeconds=5)

############################################################
# Problem 2d: dotProduct

grader.addBasicPart('2d-0-basic', lambda : grader.requireIsEqual(15, submission.sparseVectorDotProduct(collections.defaultdict(float, {'a': 5}), collections.defaultdict(float, {'b': 2, 'a': 3}))))

def randvec():
    v = collections.defaultdict(float)
    for _ in range(10):
        v[random.randint(0, 10)] = random.randint(0, 10) - 5
    return v
def test():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v1 = randvec()
            v2 = randvec()
            pred = submission.sparseVectorDotProduct(v1, v2)
            if solution_exist:
                answer = solution.sparseVectorDotProduct(v1, v2)
                yield grader.requireIsTrue(pred == answer)
            else:
                yield True
    all(get_gen())

grader.addHiddenPart('2d-1-hidden', test, 1)

############################################################
# Problem 2e: incrementSparseVector

def veceq(vec1, vec2):
    def veclen(vec):
        return sum(1 for k, v in vec.items() if v != 0)
    if veclen(vec1) != veclen(vec2):
        return False
    else:
        return all(v == vec2.get(k, 0) for k, v in vec1.items())

def test():
    v = collections.defaultdict(float, {'a': 5})
    submission.incrementSparseVector(v, 2, collections.defaultdict(float, {'b': 2, 'a': 3}))
    grader.requireIsEqual(collections.defaultdict(float, {'a': 11, 'b': 4}), v)
grader.addBasicPart('2e-0-basic', test)

def test():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v1a = randvec()
            v1b = v1a.copy()
            v2 = randvec()
            submission.incrementSparseVector(v1a, 4, v2)
            if solution_exist:
                solution.incrementSparseVector(v1b, 4, v2)
                yield grader.requireIsTrue(veceq(v1a, v1b))
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('2e-1-hidden', test, 1)

############################################################
# Problem 2f: computeMostFrequentWord

def test2f():
    grader.requireIsEqual((set(['the', 'fox']), 2), submission.computeMostFrequentWord('the quick brown fox jumps over the lazy fox'))
grader.addBasicPart('2f-0-basic', test2f)

def test2f(numTokens, numTypes):
    import random
    random.seed(SEED)
    def get_gen():
        text = ' '.join(str(random.randint(0, numTypes)) for _ in range(numTokens))
        pred = submission.computeMostFrequentWord(text)
        if solution_exist:
            answer = solution.computeMostFrequentWord(text)
            yield grader.requireIsTrue(pred == answer)
        else:
            yield True
    all(get_gen())
grader.addHiddenPart('2f-1-hidden', lambda : test2f(1000, 10), 1)
grader.addHiddenPart('2f-2-hidden', lambda : test2f(10000, 100), 1)

grader.grade()

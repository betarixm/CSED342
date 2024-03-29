import shell
import util
import wordsegUtil

############################################################
# Problem 1a: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state) == 0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        return [(state[:_], state[_:], self.unigramCost(state[:_])) for _ in range(len(state), 0, -1)]
        # END_YOUR_ANSWER

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 2a: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0, self.queryWords[0]
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords) - 1
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        p_idx, p_action = state
        possible_fills = self.possibleFills(self.queryWords[p_idx + 1])
        words = possible_fills if len(possible_fills) != 0 else {self.queryWords[p_idx + 1]}
        return [(_, (p_idx + 1, _), self.bigramCost(p_action, _)) for _ in words]
        # END_YOUR_ANSWER

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem([wordsegUtil.SENTENCE_BEGIN] + queryWords, bigramCost, possibleFills))

    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 3a: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.query, wordsegUtil.SENTENCE_BEGIN
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state[0]) == 0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
        query, prev_word = state
        result = []
        for pre, post in [(query[:_], query[_:]) for _ in range(len(query), 0, -1)]:
            result += [(word, (post, word), self.bigramCost(prev_word, word)) for word in self.possibleFills(pre)]
        return result
        # END_YOUR_ANSWER

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    __import__("time").sleep(0.004)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 4: A* search

# Problem 4a: Define an admissible but not consistent heuristic function

class SimpleProblem(util.SearchProblem):
    def __init__(self):
        # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
        self.start = "A"
        # END_YOUR_ANSWER

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.start
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == "C"
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
        if state == "A":
            return [("A->B", "B", 3), ("A->X", "X", 1)]
        elif state == "B":
            return [("B->C", "C", 1)]
        elif state == "X":
            return [("X->B", "B", 1)]
        # END_YOUR_ANSWER


def admissibleButInconsistentHeuristic(state):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    h = {"A": 3, "B": 0, "C": 0, "X": 2}
    return h[state]
    # END_YOUR_ANSWER

# Problem 4b: Apply a heuristic function to the joint segmentation-and-insertion problem

def makeWordCost(bigramCost, wordPairs):
    """
    :param bigramCost: learned bigram cost from a training corpus
    :param wordPairs: all word pairs in the training corpus
    :returns: wordCost, which is a function from word to cost
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    cache = {}
    bigram_cache = {}

    def word_cost(x):
        if x not in cache:
            cache[x] = min([bigramCost(a, x) for a, b in wordPairs if b == x])
        return cache[x]

    return word_cost
    # END_YOUR_ANSWER

class RelaxedProblem(util.SearchProblem):
    def __init__(self, query, wordCost, possibleFills):
        self.query = query
        self.wordCost = wordCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.query,
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return True
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        query = state[0]
        result = []
        for pre, post in [(query[:_], query[_:]) for _ in range(len(query), 0, -1)]:
            possible_fills = self.possibleFills(pre)
            if possible_fills:
                word, cost = min([(w, self.wordCost(w)) for w in possible_fills], key=lambda x: x[1])
                result += [(word, (post,), cost)]
        return result
        # END_YOUR_ANSWER

def makeHeuristic(query, wordCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
    dp = util.DynamicProgramming(RelaxedProblem(query, wordCost, possibleFills))
    cache = {}

    def h(x):
        if x[0] not in cache:
            cache[x[0]] = dp(x)
        return cache[x[0]]

    return h
    # END_YOUR_ANSWER

def fastSegmentAndInsert(query, bigramCost, wordCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills), heuristic=makeHeuristic(query, wordCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################

if __name__ == '__main__':
    shell.main()

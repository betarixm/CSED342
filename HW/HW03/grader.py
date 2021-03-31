#!/usr/bin/python

import graderUtil
import util
import random
import sys
import wordsegUtil

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

def test_correct(func_name, assertion=lambda pred: True, equal=lambda x, y: x == y):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(equal(pred, answer))
    return test

def test_wrong(func_name, assertion=lambda pred: True):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(pred != answer and pred is not None)
    return test

QUERIES_SEG = [
    'ThestaffofficerandPrinceAndrewmountedtheirhorsesandrodeon',
    'hellothere officerandshort erprince',
    'howdythere',
    'The staff officer and Prince Andrew mounted their horses and rode on.',
    'whatsup',
    'duduandtheprince',
    'duduandtheking',
    'withoutthecourtjester',
    'lightbulbneedschange',
    'imagineallthepeople',
    'thisisnotmybeautifulhouse',
]

QUERIES_INS = [
    'strng',
    'pls',
    'hll thr',
    'whats up',
    'dudu and the prince',
    'frog and the king',
    'ran with the queen and swam with jack',
    'light bulbs need change',
    'ffcr nd prnc ndrw',
    'ffcr nd shrt prnc',
    'ntrntnl',
    'smthng',
    'btfl',
]

QUERIES_BOTH = [
    'stff',
    'hllthr',
    'thffcrndprncndrw',
    'ThstffffcrndPrncndrwmntdthrhrssndrdn',
    'whatsup',
    'ipovercarrierpigeon',
    'aeronauticalengineering',
    'themanwiththegoldeneyeball',
    'lightbulbsneedchange',
    'internationalplease',
    'comevisitnaples',
    'somethingintheway',
    'itselementarymydearwatson',
    'itselementarymyqueen',
    'themanandthewoman',
    'nghlrdy',
    'jointmodelingworks',
    'jointmodelingworkssometimes',
    'jointmodelingsometimesworks',
    'rtfclntllgnc',
]

CORPUS = 'leo-will.txt'

_realUnigramCost, _realBigramCost, _possibleFills, _realWordPairs = None, None, None, None

def getRealCosts():
    global _realUnigramCost, _realBigramCost, _possibleFills, _realWordPairs

    if _realUnigramCost is None:
        sys.stdout.write('Training language cost functions [corpus: %s]... ' % CORPUS)
        sys.stdout.flush()

        _realUnigramCost, _realBigramCost, _realWordPairs = wordsegUtil.makeLanguageModels(CORPUS)
        _possibleFills = wordsegUtil.makeInverseRemovalDictionary(CORPUS, 'aeiou')

        print('Done!')
        print('')

    return _realUnigramCost, _realBigramCost, _possibleFills, _realWordPairs


def add_parts_1(grader, submission):
    def t_1a_1():
        def unigramCost(x):
            if x in ['and', 'two', 'three', 'word', 'words']:
                return 1.0
            else:
                return 1000.0

        grader.requireIsEqual('', submission.segmentWords('', unigramCost))
        grader.requireIsEqual('word', submission.segmentWords('word', unigramCost))
        grader.requireIsEqual('two words', submission.segmentWords('twowords', unigramCost))
        grader.requireIsEqual('and three words', submission.segmentWords('andthreewords', unigramCost))

    grader.addBasicPart('1a-1-basic', t_1a_1, maxPoints=1, maxSeconds=2)

    def t_1a_2():
        unigramCost, *_ = getRealCosts()
        grader.requireIsEqual('word', submission.segmentWords('word', unigramCost))
        grader.requireIsEqual('two words', submission.segmentWords('twowords', unigramCost))
        grader.requireIsEqual('and three words', submission.segmentWords('andthreewords', unigramCost))

    grader.addBasicPart('1a-2-basic', t_1a_2, maxPoints=1, maxSeconds=2)

    def t_1a_3():
        unigramCost, *_ = getRealCosts()

        text_list = ['pizza',  # Word seen in corpus
                     'qqqqq',  # Even long unseen words are preferred to their arbitrary segmentations
                     'z' * 100,
                     'a'  # But 'a' is a word
                     'aaaaaa'  # With an apparent crossing point at length 6->7
                     'aaaaaaa']

        for text in text_list:
            pred = submission.segmentWords(text, unigramCost)
            if solution_exist:
                answer = solution.segmentWords(text, unigramCost)
                grader.requireIsEqual(answer, pred)

    grader.addHiddenPart('1a-3-hidden', t_1a_3, maxPoints=1, maxSeconds=3)

    def t_1a_4():
        unigramCost, *_ = getRealCosts()
        for query in QUERIES_SEG:
            query = wordsegUtil.cleanLine(query)
            parts = wordsegUtil.words(query)
            pred = [submission.segmentWords(part, unigramCost) for part in parts]
            if solution_exist:
                answer = [solution.segmentWords(part, unigramCost) for part in parts]
                grader.requireIsEqual(answer, pred)

    grader.addHiddenPart('1a-4-hidden', t_1a_4, maxPoints=3, maxSeconds=3)


def add_parts_2(grader, submission):
    def t_2a_1():
        def bigramCost(a, b):
            corpus = [wordsegUtil.SENTENCE_BEGIN] + 'beam me up scotty'.split()
            if (a, b) in list(zip(corpus, corpus[1:])):
                return 1.0
            else:
                return 1000.0

        def possibleFills(x):
            fills = {
                'bm'   : set(['beam', 'bam', 'boom']),
                'm'    : set(['me', 'ma']),
                'p'    : set(['up', 'oop', 'pa', 'epe']),
                'sctty': set(['scotty']),
            }
            return fills.get(x, set())

        grader.requireIsEqual(
            '',
            submission.insertVowels([], bigramCost, possibleFills)
        )
        grader.requireIsEqual( # No fills
            'zz$z$zz',
            submission.insertVowels(['zz$z$zz'], bigramCost, possibleFills)
        )
        grader.requireIsEqual(
            'beam',
            submission.insertVowels(['bm'], bigramCost, possibleFills)
        )
        grader.requireIsEqual(
            'me up',
            submission.insertVowels(['m', 'p'], bigramCost, possibleFills)
        )
        grader.requireIsEqual(
            'beam me up scotty',
            submission.insertVowels('bm m p sctty'.split(), bigramCost, possibleFills)
        )

    grader.addBasicPart('2a-1-basic', t_2a_1, maxPoints=1, maxSeconds=2)

    def t_2a_2():
        _, bigramCost, possibleFills, *_ = getRealCosts()

        word_seq_list = [[],
                         ['zz$z$zz'],
                         [''],
                         'wld lk t hv mr lttrs'.split(),
                         'ngh lrdy'.split()]

        for word_seq in word_seq_list:
            pred = submission.insertVowels(word_seq, bigramCost, possibleFills)
            if solution_exist:
                answer = solution.insertVowels(word_seq, bigramCost, possibleFills)
                grader.requireIsEqual(answer, pred)

    grader.addHiddenPart('2a-2-hidden', t_2a_2, maxPoints=1, maxSeconds=2)

    def t_2a_3():
        SB = wordsegUtil.SENTENCE_BEGIN

        # Check for correct use of SENTENCE_BEGIN
        def bigramCost(a, b):
            if (a, b) == (SB, 'cat'): 
                return 5.0
            elif a != SB and b == 'dog':
                return 1.0
            else:
                return 1000.0
        word_seq = ['x']
        possibleFills = lambda x: set(['cat', 'dog'])
        pred = submission.insertVowels(word_seq, bigramCost, possibleFills)
        if solution_exist:
            answer = solution.insertVowels(word_seq, bigramCost, possibleFills)
            grader.requireIsEqual(answer, pred)

        # Check for non-greediness

        def bigramCost(a, b):
            # Dog over log -- a test poem by rf
            costs = {
                (SB, 'cat'):      1.0,  # Always start with cat

                ('cat', 'log'):   1.0,  # Locally prefer log
                ('cat', 'dog'):   2.0,  # rather than dog

                ('log', 'mouse'): 3.0,  # But dog would have been
                ('dog', 'mouse'): 1.0,  # better in retrospect
            }
            return costs.get((a, b), 1000.0)

        def fills(x):
            return {
                'x1': set(['cat', 'dog']),
                'x2': set(['log', 'dog', 'frog']),
                'x3': set(['mouse', 'house', 'cat'])
            }[x]

        word_seq = 'x1 x2 x3'.split()
        possibleFills = fills
        pred = submission.insertVowels(word_seq, bigramCost, possibleFills)
        if solution_exist:
            answer = solution.insertVowels(word_seq, bigramCost, possibleFills)
            grader.requireIsEqual(answer, pred)

        # Check for non-trivial long-range dependencies
        def bigramCost(a, b):
            # Dogs over logs -- another test poem by rf
            costs = {
                (SB, 'cat'):        1.0,  # Always start with cat

                ('cat', 'log1'):    1.0,  # Locally prefer log
                ('cat', 'dog1'):    2.0,  # Rather than dog

                ('log20', 'mouse'): 1.0,  # And this might even
                ('dog20', 'mouse'): 1.0,  # seem to be okay
            }
            for i in range(1, 20):       # But along the way
            #                               Dog's cost will decay
                costs[('log' + str(i), 'log' + str(i+1))] = 0.25
                costs[('dog' + str(i), 'dog' + str(i+1))] = 1.0 / float(i)
            #                               Hooray
            return costs.get((a, b), 1000.0)

        def fills(x):
            f = {
                'x0': set(['cat', 'dog']),
                'x21': set(['mouse', 'house', 'cat']),
            }
            for i in range(1, 21):
                f['x' + str(i)] = set(['log' + str(i), 'dog' + str(i), 'frog'])
            return f[x]

        word_seq = ['x' + str(i) for i in range(0, 22)]
        possibleFills = fills
        pred = submission.insertVowels(word_seq, bigramCost, possibleFills)
        if solution_exist:
            answer = solution.insertVowels(word_seq, bigramCost, possibleFills)
            grader.requireIsEqual(answer, pred)

    grader.addHiddenPart('2a-3-hidden', t_2a_3, maxPoints=1, maxSeconds=3)

    def t_2a_4():
        _, bigramCost, possibleFills, *_ = getRealCosts()
        cmp_list = []
        for query in QUERIES_INS:
            query = wordsegUtil.cleanLine(query)
            ws = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(query)]
            pred = submission.insertVowels(ws, bigramCost, possibleFills)
            if solution_exist:
                answer = solution.insertVowels(ws, bigramCost, possibleFills)
                # grader.requireIsEqual(answer, pred)
                cmp_list.append(pred == answer)
        if solution_exist:
            grader.requireIsTrue(sum(map(int, cmp_list)) / len(cmp_list) > 0.8)

    grader.addHiddenPart('2a-4-hidden', t_2a_4, maxPoints=3, maxSeconds=3)


def add_parts_3(grader, submission):
    def t_3a_1():
        def bigramCost(a, b):
            if b in ['and', 'two', 'three', 'word', 'words']:
                return 1.0
            else:
                return 1000.0

        fills_ = {
            'nd': set(['and']),
            'tw': set(['two']),
            'thr': set(['three']),
            'wrd': set(['word']),
            'wrds': set(['words']),
        }
        fills = lambda x: fills_.get(x, set())

        grader.requireIsEqual('', submission.segmentAndInsert('', bigramCost, fills))
        grader.requireIsEqual('word', submission.segmentAndInsert('wrd', bigramCost, fills))
        grader.requireIsEqual('two words', submission.segmentAndInsert('twwrds', bigramCost, fills))
        grader.requireIsEqual('and three words', submission.segmentAndInsert('ndthrwrds', bigramCost, fills))

    grader.addBasicPart('3a-1-basic', t_3a_1, maxPoints=1, maxSeconds=2)

    def t_3a_2():
        unigramCost, *_ = getRealCosts()
        bigramCost = lambda a, b: unigramCost(b)

        fills_ = {
            'nd': set(['and']),
            'tw': set(['two']),
            'thr': set(['three']),
            'wrd': set(['word']),
            'wrds': set(['words']),
        }
        fills = lambda x: fills_.get(x, set())

        grader.requireIsEqual(
            'word',
            submission.segmentAndInsert('wrd', bigramCost, fills))
        grader.requireIsEqual(
            'two words',
            submission.segmentAndInsert('twwrds', bigramCost, fills))
        grader.requireIsEqual(
            'and three words',
            submission.segmentAndInsert('ndthrwrds', bigramCost, fills))

    grader.addBasicPart('3a-2-basic', t_3a_2, maxPoints=1, maxSeconds=2)

    def t_3a_3():
        def bigramCost(a, b):
            corpus = [wordsegUtil.SENTENCE_BEGIN] + 'beam me up scotty'.split()
            if (a, b) in list(zip(corpus, corpus[1:])):
                return 1.0
            else:
                return 1000.0

        def possibleFills(x):
            fills = {
                'bm'   : set(['beam', 'bam', 'boom']),
                'm'    : set(['me', 'ma']),
                'p'    : set(['up', 'oop', 'pa', 'epe']),
                'sctty': set(['scotty']),
                'z'    : set(['ze']),
            }
            return fills.get(x, set())

        # Ensure no non-word makes it through
        text_list = ['zzzzz', 'bm', 'mp', 'bmmpsctty']
        for text in text_list:
            pred = submission.segmentAndInsert(text, bigramCost, possibleFills)
            if solution_exist:
                answer = solution.segmentAndInsert(text, bigramCost, possibleFills)
                grader.requireIsEqual(answer, pred)

    grader.addHiddenPart('3a-3-hidden', t_3a_3, maxPoints=1, maxSeconds=3)

    def t_3a_4():
        unigramCost, bigramCost, possibleFills, *_ = getRealCosts()
        smoothCost = wordsegUtil.smoothUnigramAndBigram(unigramCost, bigramCost, 0.2)
        for query in QUERIES_BOTH:
            query = wordsegUtil.cleanLine(query)
            parts = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(query)]
            pred = [submission.segmentAndInsert(part, smoothCost, possibleFills) for part in parts]
            if solution_exist:
                answer = [solution.segmentAndInsert(part, smoothCost, possibleFills) for part in parts]
                grader.requireIsEqual(answer, pred)

    grader.addHiddenPart('3a-4-hidden', t_3a_4, maxPoints=3, maxSeconds=3)

def add_parts_4(grader, submission):
    def make_t_4a(checkEndState):
        def t_4a():
            problem = submission.SimpleProblem()

            ucs = util.UniformCostSearch(verbose=0)
            ucs.solve(problem)

            astar = util.UniformCostSearch(verbose=0)
            heuristic = submission.admissibleButInconsistentHeuristic
            astar.solve(problem, heuristic)

            # check admissibility
            tmp_problem = submission.SimpleProblem()
            tmp_ucs = util.UniformCostSearch(verbose=0)
            def future(state):
                tmp_problem.startState = lambda: state
                tmp_ucs.solve(tmp_problem)
                return tmp_ucs.totalCost
            for state in astar.visitedStates:
                heuristicCost = heuristic(state)
                futureCost = future(state)
                grader.requireIsTrue(heuristicCost <= futureCost)

            # adjust heuristic. it's necessary only when h(endState) != 0
            ucsTotalCost = ucs.totalCost - heuristic(ucs.endState)

            if checkEndState:
                grader.requireIsTrue(heuristic(ucs.endState) == 0)

            # A* doesn't find minimum cost path
            grader.requireIsTrue(ucsTotalCost < astar.totalCost)
        return t_4a

    grader.addBasicPart('4a-1-basic', make_t_4a(False), maxPoints=3, maxSeconds=3)
    grader.addBasicPart('4a-2-basic', make_t_4a(True), maxPoints=1, maxSeconds=3)

    def t_4b_test():
        unigramCost, bigramCost, possibleFills, wordPairs = getRealCosts()
        smoothCost = wordsegUtil.smoothUnigramAndBigram(unigramCost, bigramCost, 0.2)
        wordCost = submission.makeWordCost(smoothCost, wordPairs)
        tm = graderUtil.TimeMeasure()

        time_cmp_list = []

        for _ in range(5):
            for query in QUERIES_BOTH:
                query = wordsegUtil.cleanLine(query)
                parts = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(query)]

                tm.check()
                pred_ucs = [submission.segmentAndInsert(part, smoothCost, possibleFills) for part in parts]
                time_sub_ucs = tm.elapsed()

                tm.check()
                pred_astar = [submission.fastSegmentAndInsert(part, smoothCost, wordCost, possibleFills) for part in parts]
                time_sub_astar = tm.elapsed()
                time_cmp_list.append(time_sub_ucs > time_sub_astar)

                grader.requireIsEqual(pred_ucs, pred_astar)

                if solution_exist:
                    tm.check()
                    answer_ucs = [solution.segmentAndInsert(part, smoothCost, possibleFills) for part in parts]
                    time_sol_astar = tm.elapsed()
                    grader.requireIsEqual(answer_ucs, pred_astar)

        # A* is faster than UCS at least in 80% of comparisons
        grader.requireIsTrue((sum(map(int, time_cmp_list)) / len(time_cmp_list)) > 0.8)

    grader.addHiddenPart('4b-1-hidden', t_4b_test, maxPoints=2, maxSeconds=8)
    grader.addHiddenPart('4b-2-hidden', t_4b_test, maxPoints=2, maxSeconds=5)

# Avoid timeouts during later non-basic parts.
getRealCosts()

add_parts_1(grader, submission)
add_parts_2(grader, submission)
add_parts_3(grader, submission)
add_parts_4(grader, submission)
grader.grade()

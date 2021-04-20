#!/usr/bin/env python

import random, util, collections
import graderUtil

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

############################################################
# Problem 1

def test_1a_1():
    mdp1 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=10, peekCost=1)
    startState = mdp1.startState()
    preBustState = (6, None, (1, 1))
    postBustState = (11, None, None)

    mdp2 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=15, peekCost=1)
    preEmptyState = (11, None, (1,0))

    # Make sure the succAndProbReward function is implemented correctly.
    tests = [
        ([((1, None, (1, 2)), 0.5, 0), ((5, None, (2, 1)), 0.5, 0)], mdp1, startState, 'Take'),
        ([((0, 0, (2, 2)), 0.5, -1), ((0, 1, (2, 2)), 0.5, -1)], mdp1, startState, 'Peek'),
        ([((0, None, None), 1, 0)], mdp1, startState, 'Quit'),
        ([((7, None, (0, 1)), 0.5, 0), ((11, None, None), 0.5, 0)], mdp1, preBustState, 'Take'),
        ([], mdp1, postBustState, 'Take'),
        ([], mdp1, postBustState, 'Peek'),
        ([], mdp1, postBustState, 'Quit'),
        ([((12, None, None), 1, 12)], mdp2, preEmptyState, 'Take')
    ]
    for gold, mdp, state, action in tests:
        if not grader.requireIsEqual(gold,
                                     mdp.succAndProbReward(state, action)):
            print('   state: {}, action: {}'.format(state, action))
grader.addBasicPart('1a-1-basic', test_1a_1, 3, description="Basic test for succAndProbReward() that covers several edge cases.")

def test_1a_2():
    def solve(BlackjackMDP):
        mdp = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                           threshold=40, peekCost=1)
        startState = mdp.startState()
        alg = util.ValueIteration()
        alg.solve(mdp, .0001)
        return alg.V[startState]

    pred = solve(submission.BlackjackMDP)
    if solution_exist:
        answer = solve(solution.BlackjackMDP)
        grader.requireIsTrue((abs(pred - answer) / answer) < 0.1)
        
grader.addHiddenPart('1a-2-hidden', test_1a_2, 2, description="Hidden test for ValueIteration. Run ValueIteration on BlackjackMDP, then test if V[startState] is correct.")

def get_test_1b(multiplier):
    if solution_exist:
        BlackjackMDP = solution.BlackjackMDP
    else:
        BlackjackMDP = submission.BlackjackMDP

    def solve(ValueIteration, *args):
        mdp = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                                    threshold=40, peekCost=1)
        startState = mdp.startState()
        alg = ValueIteration()
        alg.solve(mdp, *args)
        return alg.V[startState]

    def test_1b():
        tm = graderUtil.TimeMeasure()

        for _ in range(3):
            tm.check()
            util_vi = solve(util.ValueIteration, .00001)
            util_vi_time = tm.elapsed()

            tm.check()
            pred_dp = solve(submission.ValueIterationDP)
            pred_dp_time = tm.elapsed()

            print('VI time: {} / DP time: {}'.format(util_vi_time, pred_dp_time))
            grader.requireIsTrue((abs(pred_dp - util_vi) / util_vi) < 0.00001)
            grader.requireIsTrue(pred_dp_time * multiplier < util_vi_time)

    return test_1b

grader.addHiddenPart('1b-1-hidden', get_test_1b(2), 3, description="Hidden test for ValueIterationDP. Run ValueIterationDP on BlackjackMDP, then test if V[startState] is correct and ValueIterationDP is faster than ValueIteration.")
grader.addHiddenPart('1b-2-hidden', get_test_1b(3), 2, description="Hidden test for ValueIterationDP. Run ValueIterationDP on BlackjackMDP, then test if V[startState] is correct and ValueIterationDP is faster than ValueIteration.")

############################################################
# Problem 2

def test_2a_1():
    mdp = util.NumberLineMDP()
    rl = submission.Qlearning(mdp.actions, mdp.discount(),
                              submission.identityFeatureExtractor,
                              0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback([0, 1, 0, 1], mdp.isEnd)
    grader.requireIsEqual(0, rl.getQ(0, -1))
    grader.requireIsEqual(0, rl.getQ(0, 1))

    rl.incorporateFeedback([1, 1, 1, 2], mdp.isEnd)
    grader.requireIsEqual(0, rl.getQ(0, -1))
    grader.requireIsEqual(0, rl.getQ(0, 1))
    grader.requireIsEqual(0, rl.getQ(1, -1))
    grader.requireIsEqual(1, rl.getQ(1, 1))

    rl.incorporateFeedback([2, -1, 1, 1], mdp.isEnd)
    grader.requireIsEqual(1.9, rl.getQ(2, -1))
    grader.requireIsEqual(0, rl.getQ(2, 1))

grader.addBasicPart('2a-1-basic', test_2a_1, 3, maxSeconds=3, description="Basic test for incorporateFeedback() using NumberLineMDP.")


def test_2a_2():
    if solution_exist:
        BlackjackMDP = solution.BlackjackMDP
    else:
        BlackjackMDP = submission.BlackjackMDP
    mdp = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

    def get_policy(Qlearning):
        rl = Qlearning(mdp.actions, mdp.discount(),
                       lambda state, action: [((state, action), 1)],
                       0.2)  # 0.2 is the default epsilon
        random.seed(SEED)
        util.simulate(mdp, rl, numTrials=30000)
        rl.explorationProb = 0.0
        policy = {state: rl.getAction(state) for state in mdp.states}
        return policy

    pred = get_policy(submission.Qlearning)

    if solution_exist:
        answer = get_policy(solution.Qlearning)

        all_states = [state for state in mdp.states
                      if not mdp.isEnd(state)]
        grader.requireIsTrue((sum(int(pred[state] == answer[state])
                                  for state in all_states) / len(all_states)) > 0.95)
        
grader.addHiddenPart('2a-2-hidden', test_2a_2, 2, maxSeconds=3, description="Hidden test for incorporateFeedback(). Run Qlearning on a small MDP, then ensure that getQ returns reasonable policy.")

def test_2b_1():
    mdp = util.NumberLineMDP()
    rl = submission.SARSA(mdp.actions, mdp.discount(),
                          submission.identityFeatureExtractor,
                          0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback([0, 1, 0, 1, 1, None, None], mdp.isEnd)
    grader.requireIsEqual(0, rl.getQ(0, -1))
    grader.requireIsEqual(0, rl.getQ(0, 1))

    rl.incorporateFeedback([1, 1, 1, 2, -1, None, None], mdp.isEnd)
    grader.requireIsEqual(0, rl.getQ(0, -1))
    grader.requireIsEqual(0, rl.getQ(0, 1))
    grader.requireIsEqual(0, rl.getQ(1, -1))
    grader.requireIsEqual(1, rl.getQ(1, 1))

    rl.incorporateFeedback([2, -1, 1, 1, 1, None, None], mdp.isEnd)
    grader.requireIsEqual(1.9, rl.getQ(2, -1))
    grader.requireIsEqual(0, rl.getQ(2, 1))

grader.addBasicPart('2b-1-basic', test_2b_1, 3, maxSeconds=3, description="Basic test for incorporateFeedback() using NumberLineMDP.")


def test_2b_2():
    if solution_exist:
        BlackjackMDP = solution.BlackjackMDP
    else:
        BlackjackMDP = submission.BlackjackMDP
    mdp = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

    def get_policy(SARSA):
        rl = SARSA(mdp.actions, mdp.discount(),
                                lambda state, action: [((state, action), 1)],
                                0.2)  # 0.2 is the default epsilon
        random.seed(SEED)
        util.simulate(mdp, rl, numTrials=30000)
        rl.explorationProb = 0.0
        policy = {state: rl.getAction(state) for state in mdp.states}
        return policy

    pred = get_policy(submission.SARSA)

    if solution_exist:
        answer = get_policy(solution.SARSA)

        all_states = [state for state in mdp.states
                      if not mdp.isEnd(state)]
        grader.requireIsTrue((sum(int(pred[state] == answer[state])
                                  for state in all_states) / len(all_states)) > 0.95)
        
grader.addHiddenPart('2b-2-hidden', test_2b_2, 2, maxSeconds=3, description="Hidden test for incorporateFeedback(). Run SARSA on a small MDP, then ensure that getQ returns reasonable policy.")

def test_2c_1():
    mdp = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                  threshold=10, peekCost=1)
    rl = submission.Qlearning(mdp.actions, mdp.discount(),
                              submission.blackjackFeatureExtractor,
                              0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback([(7, None, (0, 1)), 'Quit', 7, (7, None, None)], mdp.isEnd)
    grader.requireIsEqual(28, rl.getQ((7, None, (0, 1)), 'Quit'))
    grader.requireIsEqual(7, rl.getQ((7, None, (1, 0)), 'Quit'))
    grader.requireIsEqual(14, rl.getQ((2, None, (0, 2)), 'Quit'))
    grader.requireIsEqual(0, rl.getQ((2, None, (0, 2)), 'Take'))
grader.addBasicPart('2c-1-basic', test_2c_1, 3, maxSeconds=3, description="Basic test for blackjackFeatureExtractor.  Runs Qlearning using blackjackFeatureExtractor, then checks to see that Q-values are correct.")

def test_2c_2():
    if solution_exist:
        BlackjackMDP = solution.BlackjackMDP
        Qlearning = solution.Qlearning
    else:
        BlackjackMDP = submission.BlackjackMDP
        Qlearning = submission.Qlearning

    mdp = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

    rl = Qlearning(mdp.actions, mdp.discount(),
                   submission.blackjackFeatureExtractor,
                   0.2)  # 0.2 is the default epsilon
    random.seed(SEED)
    util.simulate(mdp, rl, numTrials=30000)
    rl.explorationProb = 0.0
    rl_policy = {state: rl.getAction(state) for state in mdp.states}

    vi = util.ValueIteration()
    vi.solve(mdp)
    vi_policy = vi.pi
    
    all_states = [state for state in mdp.states
                  if not mdp.isEnd(state)]
    agreement = sum(int(rl_policy[state] == vi_policy[state])
                    for state in all_states) / len(all_states)
    grader.requireIsTrue(agreement > 0.7)
    print('Policy agreement:', agreement)

grader.addHiddenPart('2c-2-hidden', test_2c_2, 2, maxSeconds=20, description="Hidden test for incorporateFeedback(). Run Qlearning on a large MDP, then ensure that getQ returns reasonable policy.")


grader.grade()

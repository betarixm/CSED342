#!/usr/bin/env python
"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""

import random

import graderUtil
import util
import collections
import copy
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

############################################################
# Problem 0a: Simple Chain CSP

def test0a():
    solver = submission.BacktrackingSearch()
    solver.solve(submission.create_chain_csp(4))
    grader.requireIsEqual(1, solver.optimalWeight)
    grader.requireIsEqual(2, solver.numOptimalAssignments)
    grader.requireIsEqual(9, solver.numOperations)

grader.addBasicPart('0a-1-basic', test0a, 2, maxSeconds=1, description="Basic test for create_chain_csp")

def get_csp_result(csp, BacktrackingSearch=None, **kargs):
    if BacktrackingSearch is None:
        BacktrackingSearch = (solution.BacktrackingSearch if solution_exist else
                              submission.BacktrackingSearch)
    solver = BacktrackingSearch()
    solver.solve(csp, **kargs)
    return (solver.optimalWeight,
            solver.numOptimalAssignments,
            solver.numOperations)

def test0a():
    pred = get_csp_result(submission.create_chain_csp(6))
    if solution_exist:
        grader.requireIsEqual(get_csp_result(solution.create_chain_csp(6)), pred)

grader.addHiddenPart('0a-1-hidden', test0a, 1, maxSeconds=1, description="Hidden test for create_chain_csp")

############################################################
# Problem 1a: N-Queens

def test1a_1():
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(8))
    grader.requireIsEqual(1.0, nQueensSolver.optimalWeight)
    grader.requireIsEqual(92, nQueensSolver.numOptimalAssignments)
    grader.requireIsEqual(2057, nQueensSolver.numOperations)

grader.addBasicPart('1a-1-basic', test1a_1, 2, maxSeconds=1, description="Basic test for create_nqueens_csp for n=8")

def test1a_hidden(n):
    pred = get_csp_result(submission.create_nqueens_csp(n))
    if solution_exist:
        grader.requireIsEqual(get_csp_result(solution.create_nqueens_csp(n)), pred)

grader.addHiddenPart('1a-2-hidden', lambda: test1a_hidden(3), 1, maxSeconds=1, description="Test create_nqueens_csp with n=3")

grader.addHiddenPart('1a-3-hidden', lambda: (test1a_hidden(4), test1a_hidden(7)) , 1, maxSeconds=1, description="Test create_nqueens_csp with different n")

############################################################
# Problem 1b: Most constrained variable


def test1b_1():
    mcvSolver = submission.BacktrackingSearch()
    mcvSolver.solve(submission.create_nqueens_csp(8), mcv = True)
    grader.requireIsEqual(1.0, mcvSolver.optimalWeight)
    grader.requireIsEqual(92, mcvSolver.numOptimalAssignments)
    grader.requireIsEqual(1361, mcvSolver.numOperations)

grader.addBasicPart('1b-1-basic', test1b_1, 2, maxSeconds=1, description="Basic test for MCV with n-queens CSP")

def test1b_2():
    def get_csp_result_with_mcv(BacktrackingSearch):
        return get_csp_result(util.create_map_coloring_csp(), BacktrackingSearch, mcv=True)
    pred = get_csp_result_with_mcv(submission.BacktrackingSearch)
    if solution_exist:
        answer = get_csp_result_with_mcv(solution.BacktrackingSearch)
        grader.requireIsEqual(answer, pred)

grader.addHiddenPart('1b-2-hidden', test1b_2, 1, maxSeconds=1, description="Test MCV with different CSPs")

def test1b_3():
    # We will use our implementation of n-queens csp
    # mcvSolver.solve(our_nqueens_csp(8), mcv = True)
    create_nqueens_csp = (solution.create_nqueens_csp if solution_exist else
                          submission.create_nqueens_csp)
    def get_csp_result_with_mcv(BacktrackingSearch):
        return get_csp_result(create_nqueens_csp(8), BacktrackingSearch, mcv=True)
    pred = get_csp_result_with_mcv(submission.BacktrackingSearch)
    if solution_exist:
        answer = get_csp_result_with_mcv(solution.BacktrackingSearch)
        grader.requireIsEqual(answer, pred)

grader.addHiddenPart('1b-3-hidden', test1b_3, 1, maxSeconds=1, description="Test for MCV with n-queens CSP")

############################################################
# Problem 1c: Arc consistency

def test1c_1():
    acSolver = submission.BacktrackingSearch()
    acSolver.solve(submission.create_nqueens_csp(8), ac3 = True)
    grader.requireIsEqual(1.0, acSolver.optimalWeight)
    grader.requireIsEqual(92, acSolver.numOptimalAssignments)
    grader.requireIsEqual(21, acSolver.firstAssignmentNumOperations)
    grader.requireIsEqual(769, acSolver.numOperations)

grader.addBasicPart('1c-1-basic', test1c_1, 2, maxSeconds=1, description="Basic test for AC-3 with n-queens CSP")

def test1c_2():
    def get_csp_result_with_ac3(BacktrackingSearch):
        return get_csp_result(util.create_map_coloring_csp(), BacktrackingSearch, ac3=True)
    if solution_exist:
        pred = get_csp_result_with_ac3(submission.BacktrackingSearch)
        answer = get_csp_result_with_ac3(solution.BacktrackingSearch)
        grader.requireIsEqual(answer, pred)

grader.addHiddenPart('1c-2-hidden', test1c_2, 2, maxSeconds=1, description="Test AC-3 for map coloring CSP")

def test1c_3():
    # We will use our implementation of n-queens csp
    # acSolver.solve(our_nqueens_csp(8), mcv = True, ac3 = True)
    create_nqueens_csp = (solution.create_nqueens_csp if solution_exist else
                          submission.create_nqueens_csp)
    def get_csp_result_with_mcv_ac3(BacktrackingSearch):
        return get_csp_result(create_nqueens_csp(8), BacktrackingSearch, mcv=True, ac3=True)
    pred = get_csp_result_with_mcv_ac3(submission.BacktrackingSearch)
    if solution_exist:
        answer = get_csp_result_with_mcv_ac3(solution.BacktrackingSearch)
        grader.requireIsEqual(answer, pred)

grader.addHiddenPart('1c-3-hidden', test1c_3, 1, maxSeconds=1, description="Test MCV+AC-3 for n-queens CSP with n=8")

############################################################
# Problem 2a: Sum factor

def test2a_1():
    csp = util.CSP()
    csp.add_variable('A', [0, 1, 2, 3])
    csp.add_variable('B', [0, 6, 7])
    csp.add_variable('C', [0, 5])

    sumVar = submission.get_sum_variable(csp, 'sum-up-to-15', ['A', 'B', 'C'], 15)
    csp.add_unary_factor(sumVar, lambda n: n in [12, 13])
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.requireIsEqual(4, sumSolver.numOptimalAssignments)

    csp.add_unary_factor(sumVar, lambda n: n == 12)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.requireIsEqual(2, sumSolver.numOptimalAssignments)

grader.addBasicPart('2a-1-basic', test2a_1, 2, maxSeconds=1, description="Basic test for get_sum_variable")

def test2a_2():
    BacktrackingSearch = (solution.BacktrackingSearch if solution_exist else
                          submission.BacktrackingSearch)

    def get_result(get_sum_variable):
        csp = util.CSP()
        sumVar = get_sum_variable(csp, 'zero', [], 15)
        sumSolver = BacktrackingSearch()
        sumSolver.solve(csp)
        out1 = sumSolver.numOptimalAssignments

        csp = util.CSP()
        sumVar = get_sum_variable(csp, 'zero', [], 15)
        csp.add_unary_factor(sumVar, lambda n: n > 0)
        sumSolver = BacktrackingSearch()
        sumSolver.solve(csp)
        out2 = sumSolver.numOptimalAssignments

        return out1, out2

    pred = get_result(submission.get_sum_variable)
    if solution_exist:
        grader.requireIsEqual(get_result(solution.get_sum_variable), pred)

grader.addHiddenPart('2a-2-hidden', test2a_2, 1, maxSeconds=1, description="Test get_sum_variable with empty list of variables")

def test2a_3():
    def get_result(get_sum_variable):
        csp = util.CSP()
        csp.add_variable('A', [0, 1, 2])
        csp.add_variable('B', [0, 1, 2])
        csp.add_variable('C', [0, 1, 2])

        sumVar = submission.get_sum_variable(csp, 'sum-up-to-7', ['A', 'B', 'C'], 7)
        sumSolver = submission.BacktrackingSearch()
        sumSolver.solve(csp)
        out1 = sumSolver.numOptimalAssignments

        csp.add_unary_factor(sumVar, lambda n: n == 6)
        sumSolver = submission.BacktrackingSearch()
        sumSolver.solve(csp)
        out2 = sumSolver.numOptimalAssignments

        return out1, out2

    pred = get_result(submission.get_sum_variable)
    if solution_exist:
        grader.requireIsEqual(get_result(solution.get_sum_variable), pred)

grader.addHiddenPart('2a-3-hidden', test2a_3, 2, maxSeconds=1, description="Test get_sum_variable with different variables")


############################################################
# Problem 2b: Light-bulb problem

def test2b_1():
    numBulbs = 3
    numButtons = 3
    maxNumRelations = 2
    buttonSets=({0, 2}, {1, 2}, {1, 2})

    csp = submission.create_lightbulb_csp(buttonSets, numButtons)
    solver = submission.BacktrackingSearch()
    solver.solve(csp)

    pred = solver.numOptimalAssignments
    answer = 2
    grader.requireIsEqual(answer, pred)

grader.addBasicPart('2b-1-basic', test2b_1, 2, maxSeconds=1, description="Basic test for light-bulb problem")

def test2b_2():
    numBulbs = 10
    numButtons = 10
    maxNumRelations = 7
    all_buttons = list(range(numButtons))

    random.seed(SEED)
    buttonSets = tuple(set(random.sample(all_buttons, maxNumRelations))
                       for bulbIndex in range(numBulbs))

    pred = get_csp_result(submission.create_lightbulb_csp(buttonSets, numButtons))
    if solution_exist:
        answer = get_csp_result(solution.create_lightbulb_csp(buttonSets, numButtons))
        grader.requireIsEqual(answer, pred)

grader.addHiddenPart('2b-2-hidden', test2b_2, 2, maxSeconds=1, description="Test light-bulb problem arguments")


grader.grade()

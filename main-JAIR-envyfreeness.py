#!python3

"""
Simulation of existence and fairness of NDDEF allocations.

Author: Erel Segai-Halevi
Date:   2019-07
"""

import numpy as np
import pandas
from pandas import DataFrame
from pandas.tools import plotting
from itemAssignment import *
from collections import OrderedDict
from datetime import datetime

import simulations

np.random.seed(1)


def checkEnvyFreeness(prefProfile: PrefProfile):
    """
    INPUT: a preference profile.

    OUTPUT:  a vector of booleans indicating whether this profile admits various kinds of envy-free allocations.
    #
    >>> prefProfile = PrefProfile({"Alice":Pref(cardinal={6:6,5:5,4:4,3:3,2:2,1:1}), "Bob":Pref(cardinal={6:6,5:5,4:4,3:3,2:2,1:1})})
    >>> checkEnvyFreeness(prefProfile)
    (0, False, 0, 0, False, 0, 0, False, 4, 0, True, 10, 0, True, 0, 0, False, 0, 0, False)
    """
    sumFair = \
        sumNecEF = sumNecEFFair = \
        sumNDDEF = sumNDDEFFair = \
        sumPDDEF = sumPDDEFFair = \
        sumPosEF = sumPosEFFair = \
        sumWeakPDDEF = sumWeakPDDEFFair = \
        sumWeakPosEF = sumWeakPosEFFair = \
        sumABCCBA  = sumABCCBAFair  = \
        sumBaseline  = sumBaselineFair  = \
        0
    for allocation in equalPartitions(prefProfile.agents, prefProfile.items):
        isCardEF = isCardinallyEnvyFree(prefProfile, allocation)
        isNecEF = isNecessarilyEnvyFree(prefProfile, allocation)
        isNDDEF = isNDDEnvyFree(prefProfile, allocation)
        # isPDDEF = isPDDEnvyFree(prefProfile, allocation)  # We do not have an efficient algorithm for that!
        # isPosEF = isPossiblyEnvyFree(prefProfile, allocation)  # We do not have an efficient algorithm for that!
        isWeakPDDEF = isWeakPDDEnvyFree(prefProfile, allocation)  # We do not have an efficient algorithm for that!
        isWeakPosEF = isWeakPossiblyEnvyFree(prefProfile, allocation)  # We do not have an efficient algorithm for that!

        # Sanity checks:
        if isNecEF: assert isNDDEF
        if isNDDEF: assert isWeakPDDEF
        if isWeakPDDEF: assert isWeakPosEF
        # if isNDDEF: assert isPDDEF
        # if isPDDEF: assert isPosEF
        # if isPDDEF: assert isWeakPDDEF
        if isNecEF: assert isCardEF

        # Sums:
        sumFair += isCardEF
        sumNecEF += isNecEF
        sumNecEFFair += (isNecEF and isCardEF)
        sumNDDEF += isNDDEF
        sumNDDEFFair += (isNDDEF and isCardEF)
        # sumPDDEF += isPDDEF
        # sumPDDEFFair += (isPDDEF and isCardEF)
        # sumPosEF += isPosEF
        # sumPosEFFair += (isPosEF and isCardEF)
        sumWeakPDDEF += isWeakPDDEF
        sumWeakPDDEFFair += (isWeakPDDEF and isCardEF)
        sumWeakPosEF += isWeakPosEF
        sumWeakPosEFFair += (isWeakPosEF and isCardEF)

    bestItems = prefProfile.bestItems()
    isABCCBA = len(bestItems)==prefProfile.agentCount
    sumABCCBA += (sumFair if isABCCBA else 0)
    ABCCBA_allocation = findABCCBAAllocation(prefProfile)
    isABCCBAFair = isCardinallyEnvyFree(prefProfile, ABCCBA_allocation)
    sumABCCBAFair += (sumFair if (isABCCBA and isABCCBAFair) else 0)

    isBaseline = isABCCBA
    sumBaseline = sumABCCBA
    Baseline_allocation = findABCRandomAllocation(prefProfile)
    isBaselineFair = isCardinallyEnvyFree(prefProfile, Baseline_allocation)
    sumBaselineFair += (sumFair if (isBaseline and isBaselineFair) else 0)

    return (sumFair, \
            sumFair > 0, \
            sumNecEF, \
            sumNecEFFair, \
            sumNecEF > 0, \
            sumNDDEF, \
            sumNDDEFFair, \
            sumNDDEF > 0, \
            # sumPDDEF, \
            # sumPDDEFFair, \
            # sumPDDEF > 0, \
            # sumPosEF, \
            # sumPosEFFair, \
            # sumPosEF > 0, \
			sumWeakPDDEF, \
			sumWeakPDDEFFair, \
			sumWeakPDDEF > 0, \
			sumWeakPosEF, \
			sumWeakPosEFFair, \
			sumWeakPosEF > 0, \
			sumABCCBA, \
            sumABCCBAFair, \
            sumABCCBA   > 0,\
            sumBaseline, \
            sumBaselineFair, \
            sumBaseline > 0,\
            )

columnNames = (
    'Cardinally fair', 'Fair exists',
    'NecEF', 'NecEF and fair', 'NecEF exists',
    'NDDEF', 'NDDEF and fair', 'NDDEF exists',
    # 'PDDEF', 'PDDEF and fair', 'PDDEF exists',
    # 'PosEF', 'PosEF and fair', 'PosEF exists',
    'WeakPDDEF', 'WeakPDDEF and fair', 'WeakPDDEF exists',
    'WeakPosEF', 'WeakPosEF and fair', 'WeakPosEF exists',
    'ABCCBA','ABCCBA and fair', 'ABCCBA exists',
    'Baseline', 'Baseline and fair', 'Baseline exists',
)



if __name__ == "__main__":
    simulations.trace = print
    agents = (1,2,3)
    iterations = 1000
    createResults = False
    if createResults:
        filename = "temporary/" + str(datetime.now())
        (results1, results2) = simulations.simulateTwice(
            checkEnvyFreeness, columnNames, agents, iterations, filename)
    else:   # Use existing results:
        filename = "3agents-1000iters-ef"
        results1 = pandas.read_csv("results/"+filename+"-noise.csv")
        results2 = pandas.read_csv("results/"+filename+"-items.csv")


    for r in (results1, results2):
        for c in ("NecEF", "NDDEF", "WeakPosEF", "WeakPDDEF", "ABCCBA", "Baseline"):
            r[c+' precision'] = r.apply( \
                lambda row: row[c+' and fair'] / row[c], \
                axis=1)
            r[c+' precision err'] = r.apply( \
                lambda row: row[c+' and fair err'] / row[c] if row[c+' and fair'] < row[c] else 0, \
                axis=1)
            r[c+' recall'] = r.apply( \
                lambda row: row[c+' exists'], \
                axis=1)
            r[c+' recall err'] = r.apply( \
                lambda row: row[c+' exists err'], \
                axis=1)
            r[c+' F1'] = r.apply( \
                lambda row: (2*row[c+' precision']*row[c+' recall'])/(row[c+' precision']+row[c+' recall']), \
                axis=1)

    columnsForPrecision = [
        ('NecEF precision', 'k-o'),
        ('ABCCBA precision', 'c-x'),
        ('NDDEF precision', 'b-h'),
        # ('PDDEF precision', 'g-s'),
        ('WeakPDDEF precision', 'g-s'),
        ('Baseline precision', 'm-1'),
        # ('PosEF precision', 'r-v'),
		('WeakPosEF precision', 'g-s'),
	]
    columnsForRecall = [
        ('WeakPosEF recall', 'r-v'),
        ('WeakPDDEF recall', 'g-s'),
        # ('PosEF recall', 'r-v'),
        # ('PDDEF recall', 'g-s'),
        ('NDDEF recall', 'b-h'),
        ('NecEF recall', 'k-o'),
    ]
    columnsForF1 = [
        ('NecEF F1', 'k-o'),
        ('NDDEF F1', 'b-h'),
        # ('PDDEF F1', 'g-s'),
        # ('PosEF F1', 'r-v'),
        ('WeakPDDEF F1', 'g-s'),
        ('WeakPosEF F1', 'r-v'),
        ('ABCCBA F1', 'c-x'),
        ('Baseline F1', 'm-1'),
    ]
    simulations.plotResults(results1, results2, columnsForPrecision, title="precision", errorbars=True)
    simulations.plotResults(results1, results2, columnsForRecall, title="recall", errorbars=True)

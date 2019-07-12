#!python3

"""
Simulations of item assignment.

Author: Erel Segai-Halevi
Date:   2017-02
"""

import numpy as np
import pandas
from pandas import DataFrame
from pandas.tools import plotting
import matplotlib.pyplot as plt
from datetime import datetime

from itemAssignment import *
from partitions import equalPartitions
import operator
from timeit import default_timer as timer

import simulations

np.random.seed(1)


def checkProportionality(prefProfile: PrefProfile):
    """
    INPUT: a preference profile.

    OUTPUT:  a vector of booleans indicating whether this profile admits various kinds of proportional allocations.

    >>> prefProfile = PrefProfile({"Alice":Pref(cardinal={6:6,5:5,4:4,3:3,2:2,1:1}), "Bob":Pref(cardinal={6:6,5:5,4:4,3:3,2:2,1:1})})
    >>> checkSingleProfile(prefProfile)
    (False, False, 0, True, False, True, False, False, 1.0)
    """
    sumNecPropExists = \
        sumNDDPropExists = sumNDDPropExistsFair = \
        sumPDDPropExists = sumPDDPropExistsFair = \
        sumPosPropExists = sumPosPropExistsFair = 0
    for allocation in equalPartitions(prefProfile.agents, prefProfile.items):
        isNecProp = isNecessarilyProportional(prefProfile, allocation)
        isCardProp = isCardinallyProportional(prefProfile, allocation)
        isNDDProp = isNDDProportional(prefProfile, allocation)
        isPDDProp = isPDDProportional(prefProfile, allocation)
        isPosProp = isPossiblyProportional(prefProfile, allocation)

        # Sanity checks:
        if isNecProp: assert isNDDProp
        if isNDDProp: assert isPDDProp
        if isPDDProp: assert isPosProp
        if isNecProp: assert isCardProp

        # Sums:
        sumNecPropExists += isNecProp
        sumNDDPropExists += isNDDProp
        sumNDDPropExistsFair += (isNDDProp and isCardProp)
        sumPDDPropExists += isPDDProp
        sumPDDPropExistsFair += (isPDDProp and isCardProp)
        sumPosPropExists += isPosProp
        sumPosPropExistsFair += (isPosProp and isCardProp)

        if isNecProp: break  # strongest condition - no need to check further

    allocation = findABCCBAAllocation(prefProfile)
    sumABCCBANecProp = isNecessarilyProportional(prefProfile, allocation)
    sumABCCBACardProp = isCardinallyProportional(prefProfile, allocation)
    sumABCCBANDDProp = isNDDProportional(prefProfile, allocation)
    sumABCCBAPDDProp = isPDDProportional(prefProfile, allocation)
    sumABCCBAPosProp = isPossiblyProportional(prefProfile, allocation)

    return (sumNecPropExists > 0, \
            sumNDDPropExists > 0, \
            sumABCCBACardProp if sumNDDPropExists > 0 else 0, \
            sumPDDPropExists > 0, \
            sumABCCBACardProp if sumPDDPropExists > 0 else 0, \
            sumPosPropExists > 0, \
            sumABCCBACardProp if sumPosPropExists > 0 else 0, \
            sumABCCBACardProp, \
            prefProfile.countDiminishingDifferences() / prefProfile.agentCount
            )

columnNames = (
    'NecPR exists', 'NDDPR exists', 'NDDPR fair',
    'PDDPR exists', 'PDDPR fair',
    'PosPR exists', 'PosPR fair', 'ABCCBA fair', 'DD')


def checkEnvyFreeness(prefProfile:PrefProfile):
    """
    INPUT: a preference profile.
    OUTPUT:  a vector of booleans indicating whether this profile admits various kinds of envy-free allocations.
    NOT USED
    """
    sumNecPropExists = \
        sumNDDPropExists = sumNDDPropExistsFair = \
        sumPDDPropExists = sumPDDPropExistsFair = \
        sumPosPropExists  = sumPosPropExistsFair = 0
    sumNecEFExists = \
        sumNDDEFExists = sumNDDEFExistsFair = \
        sumPDDEFExists = sumPDDEFExistsFair = \
        sumPosEFExists  = sumPosEFExistsFair = 0
    for allocation in equalPartitions(prefProfile.agents, prefProfile.items):
        isNecProp  = isNecessarilyProportional(prefProfile, allocation)
        isCardProp = isCardinallyProportional(prefProfile, allocation)
        isNDDProp  = isNDDProportional(prefProfile, allocation)
        isPDDProp  = isPDDProportional(prefProfile, allocation)
        isPosProp  = isPossiblyProportional(prefProfile, allocation)

        isNecEF    = isNecessarilyEnvyFree(prefProfile, allocation)
        isCardEF   = isCardinallyEnvyFree(prefProfile, allocation)
        isNDDEF  = isNDDEnvyFree(prefProfile, allocation)
        isPDDEF  = isPDDEnvyFree(prefProfile, allocation)
        isPosEF  = isPossiblyEnvyFree(prefProfile, allocation)

        # Sanity checks:
        if isNecProp: assert isNDDProp
        if isNDDProp: assert isPDDProp
        if isPDDProp: assert isPosProp
        if isNecProp: assert isCardProp

        if isNecEF: assert isNecProp
        if isCardEF: assert isCardProp
        if isNDDEF: assert isNDDProp
        if isPDDEF: assert isPDDProp
        if isPosEF: assert isPosProp

        # Sums:
        sumNecPropExists += isNecProp
        sumNDDPropExists += isNDDProp
        sumNDDPropExistsFair += (isNDDProp and isCardProp)
        sumPDDPropExists += isPDDProp
        sumPDDPropExistsFair += (isPDDProp and isCardProp)
        sumPosPropExists += isPosProp
        sumPosPropExistsFair += (isPosProp and isCardProp)

        sumNecEFExists += isNecEF
        sumNDDEFExists += isNDDEF
        sumNDDEFExistsFair += (isNDDEF and isCardEF)
        sumPDDEFExists += isPDDEF
        sumPDDEFExistsFair += (isPDDEF and isCardEF)
        sumPosEFExists += isPosEF
        sumPosEFExistsFair += (isPosEF and isCardEF)

        if isNecEF: break  # strongest condition - no need to check further


    allocation = findABCCBAAllocation(prefProfile)
    sumABCCBANecProp  = isNecessarilyProportional(prefProfile, allocation)
    sumABCCBACardProp = isCardinallyProportional(prefProfile, allocation)
    sumABCCBANDDProp  = isNDDProportional(prefProfile, allocation)
    sumABCCBAPDDProp  = isPDDProportional(prefProfile, allocation)
    sumABCCBAPosProp  = isPossiblyProportional(prefProfile, allocation)

    return (sumNecPropExists>0, \
            sumNDDPropExists>0, \
            sumABCCBACardProp if sumNDDPropExists>0 else 0, \
            sumPDDPropExists>0, \
            sumABCCBACardProp if sumPDDPropExists>0 else 0, \
            sumPosPropExists>0, \
            sumABCCBACardProp if sumPosPropExists>0 else 0, \
            sumABCCBACardProp, \
            prefProfile.countDiminishingDifferences()/prefProfile.agentCount
            )


titleFontSize = 14
legendFontSize = 16
axesFontSize = 16

def plots(results1, results2):

    ax = plt.subplot(1, 2, 1)
    agentCount = int(results1['Agents'][0])
    iterations  = int(results1['Iterations'][0])
    itemCounts1  = int(results1['Items per agent'][0])

    results1['NDDPR fair 2'] = results1.apply ( \
        lambda row: row['NDDPR fair']/row['NDDPR exists'], \
        axis=1)

    ax.set_title("probability vs. noise, "+str(agentCount)+' agents, '+str(itemCounts1)+' items per agent',fontsize= titleFontSize, weight='bold')
    ax.set_xlabel('Noise size', fontsize=axesFontSize)

    results1.plot(x='Noise size',y='NDDPR exists', ax=ax, legend=True, style=['go-'], fontsize=axesFontSize)
    results1.plot(x='Noise size',y='NDDPR fair', ax=ax, legend=True, style=['g--'], fontsize=axesFontSize)
    results1.plot(x='Noise size',y='NecPR exists', ax=ax, legend=True, style=['r^-'], fontsize=axesFontSize)

    plt.legend(loc=0,prop={'size':legendFontSize})


    ax = plt.subplot(1, 2, 2, sharey=ax)
    agentCount = int(results2['Agents'][0])
    iterations  = int(results2['Iterations'][0])
    maxNoise  = results2['Noise size'][0]

    results2['NDDPR fair 2'] = results2.apply ( \
        lambda row: row['NDDPR fair']/row['NDDPR exists'], \
        axis=1)

    ax.set_title("probability vs. items, "+str(agentCount)+' agents, |noise|<='+str(maxNoise),fontsize= titleFontSize, weight='bold')
    ax.set_xlabel('Items per agent', fontsize=axesFontSize)

    results2.plot(x='Items per agent',y='NDDPR exists', ax=ax,  legend=True, style=['go-'], fontsize=axesFontSize)
    results2.plot(x='Items per agent',y='NDDPR fair', ax=ax, legend=True, style=['g--'], fontsize=axesFontSize)
    results2.plot(x='Items per agent',y='NecPR exists', ax=ax, legend=True, style=['r^-'], fontsize=axesFontSize)

    itemCounts = results2['Items per agent'].tolist()

    plt.legend(loc=0,prop={'size':legendFontSize})
    plt.xticks(itemCounts)

    plt.show()


if __name__ == "__main__":
    simulations.trace = print
    agents = [1,2]
    iterations = 10
    createResults = True
    if createResults:
        filename = str(datetime.now())
        (results1, results2) = simulations.simulateTwice(
            checkProportionality, columnNames, agents, iterations, filename)
    else:   # Use existing results:
        # filename = "3agents-200iters"
        # filename = "2agents-1000iters"
        filename = "2agents-1000iters-scale"
        results1 = pandas.read_csv("results/"+filename+"-noise.csv")
        results2 = pandas.read_csv("results/"+filename+"-items.csv")
    plots(results1, results2)

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
from itemAssignment import *
from collections import OrderedDict
from datetime import datetime

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
	sumFair = \
		sumNecProp = sumNecPropFair = \
		sumNDDProp = sumNDDPropFair = \
		sumPDDProp = sumPDDPropFair = \
		sumPosProp = sumPosPropFair = \
		sumABCCBA  = sumABCCBAFair  = \
		sumBaseline  = sumBaselineFair  = \
		0
	for allocation in equalPartitions(prefProfile.agents, prefProfile.items):
		isCardProp = isCardinallyProportional(prefProfile, allocation)
		isNecProp = isNecessarilyProportional(prefProfile, allocation)
		isNDDProp = isNDDProportional(prefProfile, allocation)
		isPDDProp = isPDDProportional(prefProfile, allocation)
		isPosProp = isPossiblyProportional(prefProfile, allocation)

		# Sanity checks:
		if isNecProp: assert isNDDProp
		if isNDDProp: assert isPDDProp
		if isPDDProp: assert isPosProp
		if isNecProp: assert isCardProp

		# Sums:
		sumFair += isCardProp
		sumNecProp += isNecProp
		sumNecPropFair += (isNecProp and isCardProp)
		sumNDDProp += isNDDProp
		sumNDDPropFair += (isNDDProp and isCardProp)
		sumPDDProp += isPDDProp
		sumPDDPropFair += (isPDDProp and isCardProp)
		sumPosProp += isPosProp
		sumPosPropFair += (isPosProp and isCardProp)



	bestItems = prefProfile.bestItems()
	isABCCBA = len(bestItems)==prefProfile.agentCount
	sumABCCBA += (sumFair if isABCCBA else 0)
	ABCCBA_allocation = findABCCBAAllocation(prefProfile)
	isABCCBAFair = isCardinallyProportional(prefProfile, ABCCBA_allocation)
	sumABCCBAFair += (sumFair if (isABCCBA and isABCCBAFair) else 0)

	isBaseline = isABCCBA
	sumBaseline = sumABCCBA
	Baseline_allocation = findABCRandomAllocation(prefProfile)
	isBaselineFair = isCardinallyProportional(prefProfile, Baseline_allocation)
	sumBaselineFair += (sumFair if (isBaseline and isBaselineFair) else 0)

	return (sumFair,
			sumNecProp, \
			sumNecPropFair, \
			sumNDDProp, \
			sumNDDPropFair, \
			sumPDDProp, \
			sumPDDPropFair, \
			sumPosProp, \
			sumPosPropFair, \
			sumABCCBA, \
			sumABCCBAFair, \
			sumBaseline, \
			sumBaselineFair, \
			)

columnNames = (
	'Cardinally fair',
	'NecPR', 'NecPR and fair',
	'NDDPR', 'NDDPR and fair',
	'PDDPR', 'PDDPR and fair',
	'PosPR', 'PosPR and fair',
	'ABCCBA','ABCCBA and fair',
	'Baseline', 'Baseline and fair',
)



if __name__ == "__main__":
	simulations.trace = print
	agents = [1,2,3]
	iterations = 50
	createResults = True
	if createResults:
		# filename = str(datetime.now())
		# filename = "2agents-1000iters-pr"
		filename = "3agents-50iters-pr"
		(results1, results2) = simulations.simulateTwice(
			checkProportionality, columnNames, agents, iterations, filename)
	else:   # Use existing results:
		# filename = "3agents-200iters"
		# filename = "2agents-1000iters"
		filename = "2agents-1000iters-pr"
		results1 = pandas.read_csv("results/"+filename+"-noise.csv")
		results2 = pandas.read_csv("results/"+filename+"-items.csv")


	for r in (results1, results2):
		for c in ("NecPR", "NDDPR", "PDDPR", "PosPR", "ABCCBA", "Baseline"):
			r[c+' precision'] = r.apply( \
				lambda row: row[c+' and fair'] / row[c], \
				axis=1)
			r[c+' recall'] = r.apply( \
				lambda row: row[c+' and fair'] / row['Cardinally fair'], \
				axis=1)
			r[c+' F1'] = r.apply( \
				lambda row: (2*row[c+' precision']*row[c+' recall'])/(row[c+' precision']+row[c+' recall']), \
				axis=1)

	columnsForPrecision = [
		('NecPR precision', 'k-o'),
		('NDDPR precision', 'b-h'),
		('PDDPR precision', 'g-s'),
		('PosPR precision', 'r-v'),
		('ABCCBA precision', 'c-x'),
		('Baseline precision', 'm-1'),
	]
	columnsForRecall = [
		('NecPR recall', 'k-o'),
		('NDDPR recall', 'b-h'),
		('PDDPR recall', 'g-s'),
		('PosPR recall', 'r-v'),
		('ABCCBA recall', 'c-x'),
		('Baseline recall', 'm-1'),
	]
	columnsForF1 = [
		('NecPR F1', 'k-o'),
		('NDDPR F1', 'b-h'),
		('PDDPR F1', 'g-s'),
		('PosPR F1', 'r-v'),
		('ABCCBA F1', 'c-x'),
		('Baseline F1', 'm-1'),
	]
	simulations.plotResults(results1, results2, columnsForPrecision)
	simulations.plotResults(results1, results2, columnsForRecall)
	simulations.plotResults(results1, results2, columnsForF1)

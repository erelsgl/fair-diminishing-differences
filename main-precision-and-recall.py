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
    #
    # >>> prefProfile = PrefProfile({"Alice":Pref(cardinal={6:6,5:5,4:4,3:3,2:2,1:1}), "Bob":Pref(cardinal={6:6,5:5,4:4,3:3,2:2,1:1})})
    # >>> checkProportionality(prefProfile)
    # (0, False, 0, 0, False, 0, 0, False, 4, 0, True, 10, 0, True, 0, 0, False, 0, 0, False)
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

	return (sumFair, \
			sumFair > 0, \
			sumNecProp, \
			sumNecPropFair, \
			sumNecProp > 0, \
			sumNDDProp, \
			sumNDDPropFair, \
			sumNDDProp > 0, \
			sumPDDProp, \
			sumPDDPropFair, \
			sumPDDProp > 0, \
			sumPosProp, \
			sumPosPropFair, \
			sumPosProp > 0, \
			sumABCCBA, \
			sumABCCBAFair, \
			sumABCCBA   > 0,\
			sumBaseline, \
			sumBaselineFair, \
			sumBaseline > 0,\
			)

columnNames = (
	'Cardinally fair', 'Fair exists',
	'NecPR', 'NecPR and fair', 'NecPR exists',
	'NDDPR', 'NDDPR and fair', 'NDDPR exists',
	'PDDPR', 'PDDPR and fair', 'PDDPR exists',
	'PosPR', 'PosPR and fair', 'PosPR exists',
	'ABCCBA','ABCCBA and fair', 'ABCCBA exists',
	'Baseline', 'Baseline and fair', 'Baseline exists',
)



if __name__ == "__main__":
	simulations.trace = print
	agents = (1,2,3)
	iterations = 1000
	createResults = False
	if createResults:
		# filename = str(datetime.now())
		# filename = "2agents-1000iters-pr"
		filename = "3agents-1000iters-pra"
		# filename = "3agents-50iters-pr"
		(results1, results2) = simulations.simulateTwice(
			checkProportionality, columnNames, agents, iterations, filename)
	else:   # Use existing results:
		filename = "3agents-1000iters-pr"
		# filename = "2agents-1000iters-pr"
		results1 = pandas.read_csv("results/"+filename+"-noise.csv")
		results2 = pandas.read_csv("results/"+filename+"-items.csv")


	for r in (results1, results2):
		for c in ("NecPR", "NDDPR", "PDDPR", "PosPR", "ABCCBA", "Baseline"):
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
		('NecPR precision', 'k-o'),
		('ABCCBA precision', 'c-x'),
		('NDDPR precision', 'b-h'),
		('PDDPR precision', 'g-s'),
		('Baseline precision', 'm-1'),
		('PosPR precision', 'r-v'),
	]
	columnsForRecall = [
		('PosPR recall', 'r-v'),
		# ('ABCCBA recall', 'c-x'),
		# ('Baseline recall', 'm-1'),
		('PDDPR recall', 'g-s'),
		('NDDPR recall', 'b-h'),
		('NecPR recall', 'k-o'),
	]
	columnsForF1 = [
		('NecPR F1', 'k-o'),
		('NDDPR F1', 'b-h'),
		('PDDPR F1', 'g-s'),
		('PosPR F1', 'r-v'),
		('ABCCBA F1', 'c-x'),
		('Baseline F1', 'm-1'),
	]
	simulations.plotResults(results1, results2, columnsForPrecision, title="precision", errorbars=True)
	simulations.plotResults(results1, results2, columnsForRecall, title="recall", errorbars=True)
	# simulations.plotResults(results1, results2, columnsForF1, errorbars=False)

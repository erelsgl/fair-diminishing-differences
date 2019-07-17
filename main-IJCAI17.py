#!python3

"""
Simulation of existence of NDDPR allocations.
This is the experiment reported in the IJCAI-2017 version of the paper:
https://dl.acm.org/citation.cfm?id=3171820

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

def checkProportionality(prefProfile):
	"""
	OUTPUT (bool,bool):  whether an NDDPR allocation exists for the given profile,
	and if it exists, whether it is cardinally fair.
	"""
	sumNecExists = sumNDDExists = sumNDDExistsFair = sumPDDExists = sumPDDExistsFair = sumPosExists  = sumPosExistsFair = 0
	for allocation in equalPartitions(prefProfile.agents, prefProfile.items):
		isNecProp  = isNecessarilyProportional(prefProfile, allocation)
		isCardProp = isCardinallyProportional(prefProfile, allocation)
		isNDDProp  = isNDDProportional(prefProfile, allocation)
		isPDDProp  = isPDDProportional(prefProfile, allocation)
		isPosProp  = isPossiblyProportional(prefProfile, allocation)

		# Sanity checks:
		if isNecProp: assert isNDDProp
		if isNDDProp: assert isPDDProp
		if isPDDProp: assert isPosProp
		if isNecProp: assert isCardProp
		#if isCardProp: assert isPosProp

		# Sums:
		sumNecExists += isNecProp
		sumNDDExists += isNDDProp
		sumNDDExistsFair += (isNDDProp and isCardProp)
		sumPDDExists += isPDDProp
		sumPDDExistsFair += (isPDDProp and isCardProp)
		sumPosExists += isPosProp
		sumPosExistsFair += (isPosProp and isCardProp)

		if isNecProp: break

	allocation = findABCCBAAllocation(prefProfile)
	sumABCCBANecProp  = isNecessarilyProportional(prefProfile, allocation)
	sumABCCBACardProp = isCardinallyProportional(prefProfile, allocation)
	sumABCCBANDDProp  = isNDDProportional(prefProfile, allocation)
	sumABCCBAPDDProp  = isPDDProportional(prefProfile, allocation)
	sumABCCBAPosProp  = isPossiblyProportional(prefProfile, allocation)

	return (sumNecExists>0, \
			sumNDDExists>0, \
			sumABCCBACardProp if sumNDDExists>0 else 0, \
			sumPDDExists>0, \
			sumABCCBACardProp if sumPDDExists>0 else 0, \
			sumPosExists>0, \
			sumABCCBACardProp if sumPosExists>0 else 0, \
			sumABCCBACardProp, \
			prefProfile.countDiminishingDifferences()/prefProfile.agentCount
			)


columnNames = (
	'NecPR exists', 'NDDPR exists', 'NDDPR fair',
	'PDDPR exists', 'PDDPR fair',
	'PosPR exists', 'PosPR fair', 'ABCCBA fair', 'DD')


if __name__ == "__main__":
	simulations.trace = print
	agents = [1,2]
	iterations = 10
	createResults = False
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

	columnsAndStyles = [
		('NDDPR exists', 'go-'),
		('NDDPR fair', 'g--'),
		('NecPR exists', 'r^-'),
	]
	simulations.plotResults(results1, results2, columnsAndStyles)

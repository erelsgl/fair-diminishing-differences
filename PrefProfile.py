#!python3
import numpy as  np
from Pref import Pref

class PrefProfile(object):
	"""
	Represents a set of several agents with different preferences
	Each agent has a name, and a Pref object representing its preferences.

    Author: Erel Segai-Halevi
    Date:   2017-02
	"""

	def __init__(self, mapAgentsToPrefs:dict):
		"""
		mapAgentsToPrefs: a dictionary that maps agents to Pref objects.
		"""
		self.agentsToPrefs = mapAgentsToPrefs
		self.prefs = list(mapAgentsToPrefs.values())
		self.agents = sorted(mapAgentsToPrefs.keys())
		self.items = sorted(next(iter(self.prefs)).ordinal)
		self.itemCount = len(self.items)
		self.agentCount = len(self.agents)
		if self.itemCount<1:
			raise ValueError("no items")
		if self.agentCount<1:
			raise ValueError("no agents")

	def __repr__(self):
		return self.prefs.__repr__()

	def removeItem(self, item):
		"""
		Remove the given item from all preferences in this profile.
		:param item:
		:return:
		"""
		if item in self.items:
			self.items.remove(item)
			for pref in self.prefs:
				pref.removeItem(item)
			self.itemCount -= 1
		else:
			raise ValueError("item "+str(item)+" not found")


	def countDiminishingDifferences(self):
		"""
		count the number of preferences in this profile that have the DD property
		"""
		return sum(map(Pref.isDiminishingDifferences, self.prefs))

	def bestItems(self)->set:
		"""
		returns a set of the best items of all agents.

	    >>> prefProfile = PrefProfile({"Alice":Pref([6,5,4,3,2,1]), "Bob":Pref([5,6,4,3,2,1]), "Carl":Pref([4,5,6,3,2,1])})
	    >>> sorted(prefProfile.bestItems())
	    [4, 5, 6]
	    >>> prefProfile = PrefProfile({"Alice":Pref([6,5,4,3,2,1]), "Bob":Pref([6,5,4,3,2,1]), "Carl":Pref([4,5,6,3,2,1])})
	    >>> sorted(prefProfile.bestItems())
	    [4, 6]
		"""
		return set(map(Pref.bestItem, self.prefs))


	@staticmethod
	def randomCardinal(agents:list, items:list, lowMarketValue:float, highMarketValue:float, maxNoiseSize:float):
		"""
		Create a random preference-profile with cardinal utilities.
		Randomization is uniform.
		"""
		marketValues = {
			item: np.random.uniform(lowMarketValue,highMarketValue)
			for item in items
			}
		return PrefProfile({agent:Pref.randomCardinal(marketValues, maxNoiseSize) for agent in agents})


	@staticmethod
	def randomCardinalGaussian(agents:list, items:list, lowMarketValue:float, highMarketValue:float, stddev:float):
		"""
		Create a random preference-profile with cardinal utilities.
		Randomization is Gaussian.
		"""
		marketValues = {
			item: np.random.uniform(lowMarketValue,highMarketValue)
			for item in items
			}
		return PrefProfile({agent:Pref.randomCardinalGaussian(marketValues, stddev) for agent in agents})


if __name__ == "__main__":
	import doctest
	(failures, tests) = doctest.testmod(report=True)
	print("{} failures, {} tests".format(failures, tests))

	prof1 = PrefProfile.randomCardinal(["A","B","C"], [1,2,3], 500, 1000, 100)
	print(prof1)
	print (prof1.countDiminishingDifferences())
	prof2 = PrefProfile.randomCardinalGaussian(["A","B","C"], [1,2,3], 500, 1000, 100)
	print(prof2)
	print(prof2.countDiminishingDifferences())

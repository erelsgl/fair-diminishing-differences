#!python3

"""
Utilities for conducting simulations on random utility profiles.

Author: Erel Segai-Halevi
Date:   2019-07
"""

import pandas, numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from partitions import equalPartitions
import operator
from timeit import default_timer as timer

from PrefProfile import PrefProfile
from mean_and_stderr import mean_and_stderr

trace = lambda *x: None  # To enable tracing, set trace=print


def avergeOverRandomProfiles(checkSingleProfile,
                             agents:list, items:list, lowMarketValue:float, highMarketValue:float, maxNoiseSize:float, iterations:int) -> (list,list):
    """
    Create many random utility profiles, and calculate various stats on them.

    :param checkSingleProfile: a function that takes a single PrefProfile object, and returns a vector of numbers describing it.
    :param agents: a list of agent-names.
    :param items:  a list of item-names.
    :param lowMarketValue, highMarketValue, maxNoiseSize: used for creating the random valuations.
    :param iterations: number of times to randomize.

    :return (means, stderrs):
        means  is a vector of floats, representing the average of the numbers returned for all random PrefProfiles.
        stderrs is a corresponding vector of the standard-errors.


    >>> dummyCheckSingleProfile = lambda profile: [True,False,5]
    >>> list(avergeOverRandomProfiles(dummyCheckSingleProfile, ["A","B"], ["x","y","z"], 1, 2, 0.5, 10))
    [array([ 1.,  0.,  5.]), array([ 0.,  0.,  0.])]
    """
    generator = lambda: np.array(checkSingleProfile(PrefProfile.randomCardinal(agents, items, lowMarketValue, highMarketValue, maxNoiseSize)))
    return mean_and_stderr(iterations, generator)


def simulate(checkSingleProfile, columnNames:list,
            agents:list, itemCounts:list, noiseSizes:list,
            lowMarketValue:float, highMarketValue:float, iterations:int, filename:str)->DataFrame:
    """
    Runs an experiment with random cardinal utility profiles.

    :param checkSingleProfile: a function that takes a single PrefProfile object, and returns a vector of numbers describing it.
    :param columnNames: a list of column-names. Should be of the same size as the vector returned by checkSingleProfile.

    :param agents: a list of agent-names.
    :param itemCounts: a list of different item-counts to try.
    :param noiseSizes: a list of different noise-amplitudes to try.
    :param lowMarketValue, highMarketValue: range for randomly selecting the market-value of each item.
    :param iterations: number of iterations to run randomly.
    :param filename:   name of file for saving the results. Will be created in subfolder "results/" with extension "csv".

    :return: a DataFrame with the experiment results.

    >>> pandas.set_option('display.max_columns', 500)
    >>> pandas.set_option('display.width', 500)
    >>> dummyCheckSingleProfile = lambda profile: [True,5]
    >>> dummyColumns = ["col1","col2"]
    >>> simulate(dummyCheckSingleProfile, dummyColumns, ["A","B"], [2,3,4], [0.3,0.7], 1, 2, 10, "doctest-simulation")
       Agents  Iterations  Noise size  Items per agent  col1  col2  col1 err  col2 err
    0     2.0        10.0         0.3              2.0   1.0   5.0       0.0       0.0
    1     2.0        10.0         0.3              3.0   1.0   5.0       0.0       0.0
    2     2.0        10.0         0.3              4.0   1.0   5.0       0.0       0.0
    3     2.0        10.0         0.7              2.0   1.0   5.0       0.0       0.0
    4     2.0        10.0         0.7              3.0   1.0   5.0       0.0       0.0
    5     2.0        10.0         0.7              4.0   1.0   5.0       0.0       0.0
    """
    meanColumnNames = list(columnNames)
    stderrColumnNames = [c+" err" for c in columnNames]
    results =  DataFrame(columns=['Agents', 'Iterations', 'Noise size', 'Items per agent'] + meanColumnNames + stderrColumnNames)
    agentCount = len(agents)
    for maxNoiseSize in noiseSizes:
        for itemCount in itemCounts:
            start = timer()
            trace("noise="+str(maxNoiseSize)+" items="+str(itemCount)+" file="+filename)
            (means,stderrs) = avergeOverRandomProfiles(checkSingleProfile,
                agents, range(itemCount * len(agents)),
                lowMarketValue, highMarketValue, maxNoiseSize, iterations)
            if len(means)!=len(columnNames):
                raise ValueError("checkSingleProfile returned {} values, but columnNames has {} values".format(len(means),len(columnNames)))
            results.loc[len(results)] = [agentCount, iterations, maxNoiseSize, itemCount] + list(means) + list(stderrs)
            results.to_csv("results/"+filename+".csv")
            trace("  " + str(timer() - start)+" seconds")
    return results



def simulateTwice(checkSingleProfile, columnNames:list,
                  agents:list, iterations:int, filename:str)->(DataFrame,DataFrame):
    """
    Run two simulation experiments: one with variable noise and one with variable item-count.

    :param agents:     a list of agent names.
    :param iterations: number of iterations to randomize.
    :param filename:   base filename for saving the results.
    :return: Two pandas.DataFrame objects, representing the results of two experiments:
       1. Fixed item-count and variable noise (written to file "<filename>-noise.csv"),
       2. Fixed noise and variable item-count (written to file "<filename>-items.csv").

    >>> pandas.set_option('display.max_columns', 500)
    >>> pandas.set_option('display.width', 500)
    >>> dummyCheckSingleProfile = lambda profile: [True,False,5]
    >>> dummyColumns = ["col1","col2","col3"]
    >>> (results1,results2) = simulateTwice(dummyCheckSingleProfile, dummyColumns, ["A","B"], 10, "doctest-simulation")
    >>> results1
       Agents  Iterations  Noise size  Items per agent  col1  col2  col3  col1 err  col2 err  col3 err
    0     2.0        10.0         0.1              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    1     2.0        10.0         0.2              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    2     2.0        10.0         0.3              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    3     2.0        10.0         0.4              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    4     2.0        10.0         0.5              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    5     2.0        10.0         0.6              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    6     2.0        10.0         0.7              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    7     2.0        10.0         0.8              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    8     2.0        10.0         0.9              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    9     2.0        10.0         1.0              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    >>> results2
       Agents  Iterations  Noise size  Items per agent  col1  col2  col3  col1 err  col2 err  col3 err
    0     2.0        10.0         0.5              2.0   1.0   0.0   5.0       0.0       0.0       0.0
    1     2.0        10.0         0.5              3.0   1.0   0.0   5.0       0.0       0.0       0.0
    2     2.0        10.0         0.5              4.0   1.0   0.0   5.0       0.0       0.0       0.0
    3     2.0        10.0         0.5              5.0   1.0   0.0   5.0       0.0       0.0       0.0
    4     2.0        10.0         0.5              6.0   1.0   0.0   5.0       0.0       0.0       0.0
    5     2.0        10.0         0.5              7.0   1.0   0.0   5.0       0.0       0.0       0.0
    6     2.0        10.0         0.5              8.0   1.0   0.0   5.0       0.0       0.0       0.0
    """
    agentCount = len(agents)

    fixedItemCount = 5 if agentCount==2 else 4
    results1 = simulate(checkSingleProfile, columnNames,
        agents,
        itemCounts = [fixedItemCount],
        noiseSizes=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1],

        lowMarketValue=1,
        highMarketValue=2,
        iterations = iterations,
        filename = filename+"-noise"
        )
    trace(results1)

    itemCounts = [2,3,4,5,6,7,8]  if agentCount==2 else [2,3,4,5,6]
    results2 = simulate(checkSingleProfile, columnNames,
        agents,
        itemCounts = itemCounts,
        noiseSizes=[.5],

        lowMarketValue=1,
        highMarketValue=2,
        iterations = iterations,
        filename = filename+"-items"
        )
    trace(results2)

    return (results1, results2)



titleFontSize = 14
legendFontSize = 10
axesFontSize = 16


def plotResults(results1:DataFrame, results2:DataFrame, columnsAndStyles:list, errorbars:bool=False):
    ax = plt.subplot(1, 2, 1)
    agentCount = int(results1['Agents'][0])
    iterations = int(results1['Iterations'][0])
    itemCounts1 = int(results1['Items per agent'][0])

    ax.set_title("probability vs. noise, " + str(agentCount) + ' agents, ' + str(itemCounts1) + ' items per agent',
                 fontsize=titleFontSize, weight='bold')
    ax.set_xlabel('Noise size', fontsize=axesFontSize)

    for columnName,style in columnsAndStyles:
        results1.plot(x='Noise size', y=columnName, yerr=columnName+" err" if errorbars else None,
                      ax=ax, legend=True, style=style, fontsize=axesFontSize)

    plt.legend(loc=0, prop={'size': legendFontSize})

    ax = plt.subplot(1, 2, 2, sharey=ax)
    agentCount = int(results2['Agents'][0])
    iterations = int(results2['Iterations'][0])
    maxNoise = results2['Noise size'][0]

    ax.set_title("probability vs. items, " + str(agentCount) + ' agents, |noise|<=' + str(maxNoise),
                 fontsize=titleFontSize, weight='bold')
    ax.set_xlabel('Items per agent', fontsize=axesFontSize)

    for columnName,style in columnsAndStyles:
        results2.plot(x='Items per agent', y=columnName, yerr=columnName+" err" if errorbars else None,
                      ax=ax, legend=True, style=style, fontsize=axesFontSize)

    itemCounts = results2['Items per agent'].tolist()

    plt.legend(loc=0, prop={'size': legendFontSize})
    plt.xticks(itemCounts)

    plt.show()



if __name__ == "__main__":
    import doctest
    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))

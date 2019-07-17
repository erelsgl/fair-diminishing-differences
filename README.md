# fair-diminishing-differences

Algorithms for fair allocation of items among agents with ordinal preferences,
assuming their underlying valuations have diminishing differences.
See [this paper](https://arxiv.org/abs/1705.07993) for details.

Prerequisites:

    pip3 install dicttools

The main files for running the experiments are in several files:

*  main-IJCAI17-experiment.py -- the experiment reported in the IJCAI 2017 paper.
*  main-JAIR-proportionality.py -- the experiments reported in the JAIR paper, for 2 or 3 agents and various criteria of proportionality.
*  main-JAIR-envyfreeness.py -- additional experiments, for 3 agents and various criteria of envy-freeness.

Each main file contains a variable named createResults:

* If it is True, then new experiments will be done and new result files will be created in the "results" folder.
You can set the parameters of the experiment (e.g. number of agents, number of iterations) in the main file. 
* If it is False, then results from previous experiments (stored in the results folder) will be plotted. 
You can choose the file to plot in the main file.
 

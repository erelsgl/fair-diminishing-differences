# fair-diminishing-differences

Algorithms for fair allocation of items among agents with ordinal preferences,
assuming their underlying valuations have diminishing differences.
See [this paper](https://arxiv.org/abs/1705.07993) for details.

Prerequisites:

    pip3 install dicttools

To run the experiment reported in the IJCAI 2017 paper:

    python3 main-IJCAI17-experiment.py    

To run the experiment reported in the JAIR paper:

    python3 main-precision-and-recall.py

It should create the graphs that appear in the paper.

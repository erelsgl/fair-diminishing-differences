"""
Calculating the mean and the standard error of a randomly-generated sample.

CODE REVIEW CREDITS:
* IEatBagels: https://codereview.stackexchange.com/a/224039/20684
* dfhwze: https://codereview.stackexchange.com/a/224045/20684

TEST RESULTS CREDITS:
* https://www.miniwebtool.com/standard-error-calculator/


AUTHOR: Erel Segal-Halevi
SINCE:  2019-07-13
"""

import numpy as np

def mean_and_stderr(num_of_iterations:int, instance_generator) -> (float,float):
    """
    Calculate the mean and standard error of the given generator.

    :param instance_generator: a function that accepts no parameters,
             and returns either a float or a numpy array.
    :param num_of_iterations: number of times to run the instance_generator.

    :return (mean, standard_error)

    Test on a degenerate (constant) generator of numbers:
    >>> def five(): return 5
    >>> mean_and_stderr(100, five)
    (5.0, 0.0)

    Test on consecutive numbers
    >>> def iota(): iota.i+=1; return iota.i
    >>> iota.i = 0
    >>> mean_and_stderr(3, iota)   # 1,2,3
    (2.0, 0.57735026918962584)
    >>> mean_and_stderr(5, iota)   # 4,5,6,7,8
    (6.0, 0.70710678118654757)

    Test on a degenerate (constant) generator of vectors:
    >>> generator = lambda: np.array([1,2,3])
    >>> mean_and_stderr(100, generator)
    (array([ 1.,  2.,  3.]), array([ 0.,  0.,  0.]))
    """
    sum = sumSquares = None
    for i in range(num_of_iterations):
        x_i = instance_generator()
        if sum is None:
            sum = x_i
            sumSquares = (x_i*x_i)
        else:
            sum += x_i
            sumSquares += (x_i * x_i)
    mean = sum / num_of_iterations
    population_variance = sumSquares / num_of_iterations - (mean*mean)
    sample_variance = population_variance*num_of_iterations/(num_of_iterations-1)
    stderr = np.sqrt(sample_variance / num_of_iterations)
    return (mean,stderr)



if __name__=="__main__":
    generate_uniformly_random_number = np.random.random
    print(mean_and_stderr(10, generate_uniformly_random_number))
    # Typical output: (0.5863703739913031, 0.026898107452102943)
    print(mean_and_stderr(1000, generate_uniformly_random_number))
    # Typical output: (0.514204422858358, 0.0002934476865378269)

    generate_uniformly_random_vector = lambda: np.random.random(3)
    print(mean_and_stderr(10, generate_uniformly_random_vector))
    # Typical output: (array([ 0.53731682,  0.6284966 ,  0.48811251]), array([ 0.02897111,  0.0262977 ,  0.03192519]))
    print(mean_and_stderr(1000, generate_uniformly_random_vector))
    # Typical output: (array([ 0.50520085,  0.49944188,  0.50034895]), array([ 0.00028528,  0.00028707,  0.00029089]))



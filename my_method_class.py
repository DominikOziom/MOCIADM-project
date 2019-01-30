# -*- coding: utf-8 -*-

import numpy as np
import operator
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

class SyntheticOverSampler:
    def __init__(self, distrib = "Gauss"):
        self.distribution = distrib # Gauss, Normal etc
        self.n_samples = 0



    def fit_sample(self, X, y):
        self.n_samples = self.returnNSamplesToGenerate(y)
        
        min_X, min_y = self.returnMinoritySamples(X, y)
        
        new_X = []
        new_y = []
        solutionArray = []
        
        for i in range(len(X[0])):
            column = min_X[:,i]
            solutionArray.append(self.generateSyntheticSamplesForColumn(column))
        
        for sampl in range(len(solutionArray[0])):
            sample = []
            for attr in range(len(solutionArray)):
                sample.append(solutionArray[attr][sampl])
            new_X.append(sample)
            new_y.append('1')
        final_y = np.asarray(new_y)
        final_X = np.asarray(new_X)

        res_X = np.concatenate((X, final_X), axis=0)
        res_y = np.concatenate((y, final_y), axis=0)

        return res_X, res_y
    
    
    
    
    def returnNSamplesToGenerate(self, y):
        """ Calculate how many samples to generate in order to have
            a balanced set
        """
        cnt = Counter()
        for word in y:
            cnt[word] += 1
        for i in cnt:
            if i == '0':
                maj = cnt[i]
            if i == '1':
                minor = cnt[i]
        result = maj - minor
        return result
    
    def returnMostDensedArea(self, column):
        """ returns value from array which lays in the most densely populated area
            and borders of this area in each direction from center point
        """
        column.sort()
        histogram, bin_edges = np.histogram(column, bins=15, range=None, weights=None, density=False)
        index, val = max(enumerate(histogram), key=operator.itemgetter(1))
        center = (bin_edges[index+1] + bin_edges[index])/2.0
        left = column[0]
        right = column[len(column)-1]
        return left, center, right
    
    def returnMinoritySamples(self, X, y):
        """ creates array populated only by samples from minority class
            returns new X and y
        """
        min_X = X
        min_y = y
        for i in range(len(y)-1, -1, -1):
            if y[i] =='0':
                min_y = np.delete(min_y, (i), axis=0)
                min_X = np.delete(min_X, (i), axis=0)
        return min_X, min_y
    
    
    def get_truncated_normal(self, mean, sd, low, upp):
        """ simple fcn for generating values with gaussian distribution
            in a given ranges of area
        """
        return truncnorm((low - mean) / sd,
                         (upp - mean) / sd,
                         loc=mean,
                         scale=sd)
    
    def generateSyntheticSamplesForColumn(self, column):
        """ Uses gaussian distribution and truncation to generate samples
            from certain range determined by column values.
            Returns array with generated values
        """
        if self.CheckIfColumnIsNotZeros(column):
            left, center, right = self.returnMostDensedArea(column)
            generator = self.get_truncated_normal(mean=center, sd=0.12, low=left, upp=right)
            new_samples = generator.rvs(self.n_samples)
        else:
            new_samples = np.zeros((self.n_samples), dtype=float)
        """
        count, bins, ignored = plt.hist(col, 30, density=True)
        plt.show()
        count, bins, ignored = plt.hist(new_samples, 30, density=True)
        plt.show()
        print(center)
        """
        return new_samples

    def CheckIfColumnIsNotZeros(self, column):
        result = False
        for elem in column:
            if elem != 0:
                result = True
        return result
        
        
        


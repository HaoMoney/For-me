#!/usr/bin/env python  
# -*- coding:utf-8 -*-

"""
/***************************************************************************
 * 
 * Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/

/**
 * @File:   sample.py
 * @Author: zhanliqing(zhanliqing@baidu.com)
 * @Time:   2018/8/23
 **/
"""
from __future__ import division
import random
import sys


class Reservoir(object):
    """
        水库采样
    """

    def __init__(self, k):
        """
        :param k: 采样多少个数据
        """
        self.k = k

    def sample(self, iter):
        """
            采样
        :param iter:
        :return:
        """

        idx = 1
        samples = []
        for line in iter:
            if idx <= self.k:
                samples.append(line)
            else:
                i = random.randint(0, sys.maxint) % idx
                if i < self.k:
                    samples[i] = line
            idx += 1
        return samples


class WeightedSampling(object):
    """
        带权采样
    """

    def __init__(self, samples, weights):
        """
            初始化
        :param samples: 待采样样本
        :param weights: 样本权重
        """
        self.weights = weights
        self.samples = samples
        self.size = len(self.samples)

    def sample(self, k, filter_sample):
        """
            采样
        :param k: 采样个数
        :param filter_sample : 过滤样本
        :return: 采样数据
        """
        max_weight = max(self.weights)
        p = map(lambda x: x / max_weight, self.weights)

        idx = 0
        sample = []
        while True:
            if idx >= k:
                return sample
            sample_idx = random.randint(0, self.size - 1)
            pr = random.random()
            if pr <= p[sample_idx]:
                if self.samples[sample_idx] in filter_sample:
                    continue
                sample.append(self.samples[sample_idx])
                idx += 1


if __name__ == '__main__':
    samples = [1, 2, 3, 4, 5, 6]
    weights = [0.001, 0.05, 0.15, 0.25, 0.1, 0.35]
    ws = WeightedSampling(samples, weights)
    s = ws.sample(100000)
    import pandas as pd

    sd = pd.Series(s)
    print(sd.value_counts() / sd.count())

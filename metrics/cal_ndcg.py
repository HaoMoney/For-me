#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author  : haolanchen
# @Email   : haolanchen@tencent.com
from collections import defaultdict
import sys
from sklearn.metrics import average_precision_score
import math
import numpy as np


def find_dcg(element_list):
    """
    Discounted Cumulative Gain (DCG)
    The definition of DCG can be found in this paper:
        Azzah Al-Maskari, Mark Sanderson, and Paul Clough. 2007.
        "The relationship between IR effectiveness measures and user satisfaction."
    Parameters:
        element_list - a list of ranks Ex: [5,4,2,2,1]
    Returns:
        score
    """
    score = 0.0
    for order, rank in enumerate(element_list):
        score += float(rank)/math.log((order+2))
    return score


def find_ndcg(label, prediction):
    """
    Normalized Discounted Cumulative Gain (nDCG)
    Normalized version of DCG:
        nDCG = DCG(prediction)/DCG(label)
    Parameters:
        label   - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        prediction  - a proposed ordering Ex: [5,2,2,3,1]
    Returns:
        ndcg_score  - normalized score
    """
    prediction_dcg = find_dcg(prediction)
    label_dcg = find_dcg(label)
    return prediction_dcg/label_dcg if label_dcg != 0 else 0.0


def find_unified_ndcg(label, prediction, uni_func=lambda x: x):
    """
    Normalized Discounted Cumulative Gain (nDCG)
    Normalized version of DCG:
        nDCG = DCG( label(prediction) ) / DCG( label )
    Parameters:
        label   - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        prediction  - a proposed ordering Ex: [50,20,20,30,1]
    Returns:
        ndcg_score  - normalized score
    """
    # index_pred_list = enumerate(prediction)
    # sorted_index_pred_list = sorted(index_pred_list, key=lambda x: x[1] * -1)
    sorted_label_list = sorted(label, reverse=True)
    pred_label_list = sorted(zip(prediction, label), key=lambda x:x[0], reverse=True)
    label_list = [l for pred, l in pred_label_list]
    # print(sorted_label_list)
    # print(label_list)
    return find_ndcg(sorted_label_list, label_list)


def find_precision_k(label, prediction, k):
    """
    Precision at k
    This measure is similar to precision but takes into account first k elements
    Description label:
        Kishida, Kazuaki. "Property of average precision and its generalization:
        An examination of evaluation indicator for information retrieval experiments."
        Tokyo, Japan: National Institute of Informatics, 2005.
    Parameters:
        label   - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        prediction  - a proposed ordering Ex: [5,2,2,3,1]
        k           - a number of top element to consider
    Returns:
        precision   - a score
    """
    precision = 0.0
    relevant = 0.0
    for i, value in enumerate(prediction[:k]):
        if value == label[i]:
            relevant += 1.0
    precision = relevant/k

    return precision


def find_precision(label, prediction):
    """
    Presision
    Description label:
        Kishida, Kazuaki. "Property of average precision and its generalization:
        An examination of evaluation indicator for information retrieval experiments."
        Tokyo, Japan: National Institute of Informatics, 2005.
    Parameters:
        label    - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        prediction   - a proposed ordering Ex: [5,2,2,3,1]
    Returns:
        precision    - a score
    """

    return find_precision_k(label, prediction, len(label))


def find_average_precision(label, prediction):
    """
    Average Precision
    Description label:
        Kishida, Kazuaki. "Property of average precision and its generalization:
        An examination of evaluation indicator for information retrieval experiments."
        Tokyo, Japan: National Institute of Informatics, 2005.
    Parameters:
        label    - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        prediction   - a proposed ordering Ex: [5,2,2,3,1]
    Returns:
        precision    - a score
    """

    s_total = sum([find_precision_k(label, prediction, k+1) for k in \
                   range(len(label))])

    return s_total/len(label)


def _order_lists(label, prediction):
    """
    Maps and orders both lists. Ex: ref:[2,5,1,1] and hyp:[2,2,3,1] =>
                                     ref:[5,2,1,1] and hyp:[1,2,5,1]
    """
    pair_ref_list = sorted([x for x in enumerate(label)], key=lambda x: x[1] * -1)
    mapped_hyp_list = [prediction[x[0]] for x in pair_ref_list]

    return [x[1] for x in pair_ref_list], mapped_hyp_list


def find_rankdcg(label, prediction):
    """
    RankDCG - modified version of well known DCG measure.
    This measure was designed to work with ties and non-normal rank distribution.
    Description label:
    RankDCG is described in this paper:
    "RankDCG: Rank-Ordering Evaluation Measure," Denys Katerenchuk, Andrew Rosenberg
    http://www.dk-lab.com/wp-content/uploads/2014/07/RankDCG.pdf
    Cost function: relative_rank(i)/reversed_rel_rank(i)
    Params:
        label_list - list: original list with correct user ranks
        prediction_list - list: predicted user ranks
    Returns:
        score - double: evaluation score
    """

    #Ordering to avoid bias with majority class output
    label_list, prediction_list = _order_lists(label, prediction)

    ordered_list = label_list[:] # creating ordered list
    ordered_list.sort(reverse=True)

    high_rank = float(len(set(label_list))) # max rank
    reverse_rank = 1.0            # min score (reversed rank)
    relative_rank_list = [high_rank]
    reverse_rank_list = [reverse_rank]

    for index, rank in enumerate(ordered_list[:-1]):
        if ordered_list[index+1] != rank:
            high_rank -= 1.0
            reverse_rank += 1.0
        relative_rank_list.append(high_rank)
        reverse_rank_list.append(reverse_rank)

    # map real rank to relative rank
    label_pair_list = [x for x in enumerate(label_list)]
    sorted_label_pairs = sorted(label_pair_list, key=lambda p: p[1], \
                                    reverse=True)
    rel_rank_label_list = [0] * len(label_list)
    for position, rel_rank in enumerate(relative_rank_list):
        rel_rank_label_list[sorted_label_pairs[position][0]] = rel_rank

    # computing max/min values (needed for normalization)
    max_score = sum([rank/reverse_rank_list[index] for index, rank \
                     in enumerate(relative_rank_list)])
    min_score = sum([rank/reverse_rank_list[index] for index, rank \
                     in enumerate(reversed(relative_rank_list))])

    # computing and mapping prediction to label
    prediction_pair_list = [x for x in enumerate(prediction_list)]
    sorted_prediction_pairs = sorted(prediction_pair_list, \
                                     key=lambda p: p[1], reverse=True)
    eval_score = sum([rel_rank_label_list[pair[0]] / reverse_rank_list[index] \
                      for index, pair in enumerate(sorted_prediction_pairs)])

    return (eval_score - min_score) / (max_score - min_score) if max_score > min_score else -1.0

def auto_balanced_weighted_mAP(label_mat, pred_mat):
    '''
    :param label_mat: [ ( <weight of query0>, np.array([ <labels<int 0,1> between query0 and label01, label02, label03...> ) ]) ]
    :param pred_mat: [ np.array([ <scores<float between 0.0 and 1.0> between query0 and label01, label02, label03...> ]) ]
    :return: weighted mean average precision
    '''
    assert len(label_mat) == len(pred_mat)
    weighted_score_sum = 0.0
    weight_sum = 0.0
    for label_tuple, pred_arr in zip(label_mat, pred_mat):
        (query_weight, label_arr) = label_tuple
        assert len(label_arr) == len(pred_arr)
        weight_sum += query_weight
        weighted_score_sum += average_precision_score(
            label_arr, pred_arr) * query_weight
    if 0 == weight_sum:
        return 0.0
    else:
        return weighted_score_sum / weight_sum


def unit_test():
    # test case 1
    label_mat = [(1.0, np.array([0, 1, 0, 1]))]
    pred_mat = [np.array([0.1, 0.4, 0.35, 0.8])]
    print('test case 1: {}'.format(
        auto_balanced_weighted_mAP(label_mat, pred_mat)))
    # test case 2
    label_mat = [(1.0, np.array([1, 1, 0, 0, 1, 1, 0, 0]))]
    pred_mat = [np.array([0.1, 0.4, 0.35, 0.8, 0.1, 0.4, 0.35, 0.8])]
    print('test case 2: {}'.format(
        auto_balanced_weighted_mAP(label_mat, pred_mat)))
    # test case 3
    label_mat = [(1.0, np.array([1, 1, 0, 0])), (10.0, np.array([0, 0, 1, 1]))]
    pred_mat = [np.array([0.1, 0.4, 0.35, 0.8]),
                np.array([0.1, 0.4, 0.35, 0.8])]
    print('test case 3: {}'.format(
        auto_balanced_weighted_mAP(label_mat, pred_mat)))
    # test case 4
    label_mat = [(1.0, np.array([1, 0, 0, 0, 0, 0]))]
    pred_mat = [np.array([0.4477845, 0.518229, 0.47325285, 0.5123511, 0.4476585, 0.430946])]
    print('test case 4: {}'.format(
        auto_balanced_weighted_mAP(label_mat, pred_mat)))
    # test case 5 for rankndcg
    label_list = [1,2,2,1,1,1,1]
    pred_list = [0.61,0.79,0.84,0.72,0.91,0.8,0.88]
    print('test case 5: {}'.format(
        find_rankdcg(label_list, pred_list)))
    # test case 6 for unified ndcg
    label_list = [5,4,3,2,1]
    pred_list = [0.5,0.4,3,0.2,9]
    print('test case 6: {}'.format(
        find_unified_ndcg(label_list, pred_list)))
    # test case 7 for unified ndcg
    label_list = [400.12, 2.65, 2.45, 2.0, 1.41]
    pred_list = [0.88, 0.89, 0.89, 0.82, 0.85]
    print('test case 7: {}'.format(
        find_unified_ndcg(label_list, pred_list)))
    # test case 8 for unified ndcg
    label_list = [1,1,1,1,2,2,2,1]
    pred_list = [0.59,0.81,0.82,0.82,0.92,0.87,0.87,0.81]
    print('test case 8: {}'.format(
        find_unified_ndcg(label_list, pred_list)))



if '__main__' == __name__:
	old_ans = []
	new_ans = []
	for item in sys.stdin:
		i = 0
		old_test_y = list()
		old_predict_y = list()
		new_test_y = list()
		new_predict_y = list()
		item = item.strip()
		if i ==0 :
			i = i + 1
			#continue
		if item:
			item  = item.split('\t')
			if len(item) == 4:
				old_tags = item[2]
				s_old_tags = item[2].split(",")
				for tmp in s_old_tags:
					tmp = tmp.split(":")
					old_test_y.append(int(tmp[2]))
					old_predict_y.append(float(tmp[1]))
				old_ndcg = find_rankdcg(old_test_y,old_predict_y)
				old_ans.append(old_ndcg)
				new_tags = item[3]
				s_new_tags = item[3].split(",")
				for tmp in s_new_tags:
					tmp = tmp.split(":")
					new_test_y.append(int(tmp[2]))
					new_predict_y.append(float(tmp[1]))
				new_ndcg = find_rankdcg(new_test_y,new_predict_y)
				new_ans.append(new_ndcg)
	i = i + 1
	print(np.mean(old_ans))			
	print(np.mean(new_ans))			

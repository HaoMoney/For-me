# -*- coding: utf8 -*-
'''
@author: visionzhong
@date: 2019-06-21
@func:
	找到最优F1
'''
import sys

from sklearn.metrics import f1_score, precision_score, recall_score

def find_f1(infile, k, region):
	'''
	Args:
		infile: 文件名，格式：以\t分割，score\tlabel
		k：枚举范围，[0, k]，与region一起使用
		region：最小粒度
	Return:

	'''
	ground_truth = []
	predict_score = []
	with open(infile, encoding='utf8') as fin:
		for line in fin:
			score, label = line.strip().split('\t')
			score = float(score.strip())
			label = int(label.strip())
			ground_truth.append(label)
			predict_score.append(score)
	best_k, best_score = 0, 0
	best_precision, best_recall = 0, 0
	for k in range(k):
		k = k*region
		predict_label = []
		for score in predict_score:
			predict_label.append(int(score>k))
		f1 = f1_score(ground_truth, predict_label)
		precision = precision_score(ground_truth, predict_label)
		recall = recall_score(ground_truth, predict_label)
		if k == 1.85:
			print(k, f1, precision, recall)
		if f1 > best_score:
			best_k = k
			best_score = f1
			best_precision = precision
			best_recall = recall
	print(best_k, best_score, best_precision, best_recall)


if __name__ == '__main__':
	# 样例
	infile = sys.argv[1] 
	find_f1(infile, 2000, 0.01)

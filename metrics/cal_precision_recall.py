#coding=utf-8
import numpy as np
import sys
def AP(ranked_list, ground_truth):
	#Compute the average precision (AP) of a list of ranked items
	
	hits = 0
	sum_precs = 0
	for n in range(len(ranked_list)):
		if ranked_list[n] in ground_truth:
			hits += 1
			sum_precs += hits / (n + 1.0)
	if hits > 0:
		return sum_precs / len(ground_truth)
	else:
		return 0
def precision(ranked_list, ground_truth):
	comm_set = set(ranked_list[:len(ground_truth)]) & set(ground_truth)
	return len(comm_set)/float(len(ranked_list))
def recall(ranked_list, ground_truth):
	comm_set = set(ranked_list[:len(ground_truth)]) & set(ground_truth)
	return len(comm_set)/float(len(ground_truth))
if __name__ == "__main__":
	truth_dict = {}
	online_dict = {}
	demo_dict = {}
	with open("ground_truth_tmp_merged","r") as f:
		data = f.readlines()
		for line in data:
			line = line.strip()
			s_line = line.split("\t")
			tmp = []
			for items in s_line[2:]:
				s_items = items.split(",")
				tmp.append(s_items[0])
			truth_dict[s_line[0]] = tmp
	with open("cmp_data_1000","r") as f:
		data = f.readlines()
		for line in data:
			line = line.strip()
			s_line = line.split("\t")
			if len(s_line) < 4:
				continue
			online_tags = s_line[2].split(",")
			demo_tags = s_line[3].split(",")
			online_tmp = []
			for item in online_tags:
				s_item = item.split(":")
				#if float(s_item[1]) > 0.4:
				#	online_tmp.append(s_item[0])
				online_tmp.append(s_item[0])
			online_dict[s_line[0]] = online_tmp
			demo_tmp = []
			for item in demo_tags:
				s_item = item.split(":")
				#if float(s_item[1]) > 0.4:
				#	demo_tmp.append(s_item[0])
				demo_tmp.append(s_item[0])
			demo_dict[s_line[0]] = demo_tmp
	online_sum = []
	demo_sum = []
	online_prec = []
	demo_prec = []
	online_recall = []
	demo_recall = []
	for url in truth_dict:
		if url in online_dict.keys() and url in demo_dict.keys():
			online_sum.append(AP(online_dict[url],truth_dict[url]))
			demo_sum.append(AP(demo_dict[url],truth_dict[url]))
			online_prec.append(precision(online_dict[url],truth_dict[url]))
			demo_prec.append(precision(demo_dict[url],truth_dict[url]))
			online_recall.append(recall(online_dict[url],truth_dict[url]))
			demo_recall.append(recall(demo_dict[url],truth_dict[url]))
	print(np.mean(online_sum))
	print(np.mean(demo_sum))
	print(np.mean(online_prec))
	print(np.mean(demo_prec))
	print(np.mean(online_recall))
	print(np.mean(demo_recall))

import numpy as np
import heapq
import sys
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import tensorflow as tf

dataParams = './PreTrainedNetParams/'
#Cifar10dat = './Data/'

probs = np.load(dataParams+'class_probs_test_cifar10.npy')
feas = np.load(dataParams+'feats_test_cifar10.npy')
(_, Y_train), (_, orig_classes) = tf.keras.datasets.cifar10.load_data()
dists = pdist(feas,'euclidean')
dists = squareform(dists)

def find_outliers_from_class(n_outlier,out_cls,base_cls):
	n = feas.shape[0]
	classes = np.array(orig_classes).reshape([-1])#.reshape(len(orig_classes),1) #np.argmax(orig_classes,1) 
	classified_classes = np.argmax(probs, axis=1)
	sel_inds_class = np.argwhere(classes==out_cls)
	# print "sel_inds_class", sel_inds_class
	sel_inds_pred = np.argwhere(classified_classes==out_cls)
	# print "sel_inds_pred", sel_inds_pred
	sel_inds = np.intersect1d(sel_inds_pred,sel_inds_class)
	# print"where they are equal:",sel_inds
	# sel_inds = np.where(np.logical_and(classes==out_cls , classified_classes==out_cls))[0]
	# print set(list(sel_inds))
	sorted_ids = np.argsort(probs[sel_inds,:].max(1))
	sel_inds = sel_inds[sorted_ids]
	outlier_ids = sel_inds[0:n_outlier]

	poisson_ids = []
	poisson_dists = []
	for cid in outlier_ids:
		min_dist,min_id = 1e10,-1
		for i in range(n):
			if(i==cid): continue
			cur_class = classes[i]
			if cur_class == out_cls: continue
			if cur_class != base_cls: continue
			cur_dist = dists[cid][i]
			if cur_dist<min_dist:
				min_dist = cur_dist
				min_id = i
		poisson_ids.append(min_id)
		poisson_dists.append(min_dist)

	print orig_classes[outlier_ids]
	print orig_classes[poisson_ids]
	print('Outlier ids: {}'.format(outlier_ids))
	print('Poisson ids: {}'.format(poisson_ids))
	print('Poisson dists: {}'.format(poisson_dists))

	return outlier_ids,poisson_ids, poisson_dists



if __name__=="__main__":
	outlier_ids,poisson_ids,poisson_dists = find_outliers_from_class(10,2,0)
	print('Outlier ids: {}'.format(outlier_ids))
	print('Poisson ids: {}'.format(poisson_ids))
	print('Poisson dists: {}'.format(poisson_dists))



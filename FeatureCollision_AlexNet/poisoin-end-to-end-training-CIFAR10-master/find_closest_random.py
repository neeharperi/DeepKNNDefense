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
#orig_classes = np.load(Cifar10dat+'Y_test.npy')
(_, _), (_, orig_classes) = tf.keras.datasets.cifar10.load_data()
# dists = pdist(feas,'euclidean')
# dists = squareform(dists)

def find_outliers_from_class_random(n_outlier,out_cls,base_cls,num_poisons, seed=None, allBasesSame=False, randomTargets=False):
	if allBasesSame: #if all bases should be the same, then fix the seed so that the experiments are consistent
		seed = 123
	np.random.seed(seed)
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

	possible_poison_ids = np.argwhere(classes==base_cls)
	possible_poison_ids = possible_poison_ids.ravel()
	# exit(0)
	poisson_ids = []
	poisson_dists = []

	np.random.shuffle(possible_poison_ids)
	for cid in outlier_ids:
		#if all bases are not forced to be the same, then shuffle the ids. Otherwise, skip the shuffling
		if not allBasesSame:
			np.random.shuffle(possible_poison_ids)
		these_random_poisons = np.copy(possible_poison_ids[:num_poisons])
		poisson_ids.append(list(these_random_poisons))

	if randomTargets: #if the targets should be radnom, shuffle the sel inds and take the outlier id
		np.random.shuffle(sel_inds)
		outlier_ids = sel_inds[0:n_outlier]



	print('Outlier ids: {}'.format(outlier_ids))
	print('Poisson ids: {}'.format(poisson_ids))
	# print('Poisson dists: {}'.format(poisson_dists))

	return outlier_ids,poisson_ids, poisson_dists



if __name__=="__main__":
	outlier_ids,poisson_ids,poisson_dists = find_outliers_from_class_random(2,0,6,3, allBasesSame=True)
	print('Outlier ids: {}'.format(outlier_ids))
	print('Poisson ids: {}'.format(poisson_ids))
	print('Poisson dists: {}'.format(poisson_dists))



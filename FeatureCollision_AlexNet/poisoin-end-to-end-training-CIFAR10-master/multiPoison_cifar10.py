"""
this is a script which tries poisoning attack on cifar 10
It makes multiple poisons
It re-uses previous poisons to save time in terms of poison making
"""
import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import math
import time
from scipy import misc
from utils_cifar10_for_multiPoison import *
from find_closest_random import find_outliers_from_class_random
from find_closest_3 import find_outliers_from_class
import csv


randomTargets = True
MultiBase = True 
targClass = 2
baseClass = 5
n_outlier = 30

opacs = [30]#,50.,80.]
poisons = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

Max_Number_of_poisons = max(poisons) #these poisons are all for the same target and starting from the same base --- they are made by adding some small random noise to the target image and moving towards its direction

#####################################################################################
#						load the data and get thigs set up
#####################################################################################
#load the data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.int32).flatten()
X_test = X_test.astype(np.float32)
Y_test = Y_test.astype(np.int32).flatten()
print('done loading data')


#####################################################################################
#						setting things up for saving
#####################################################################################
target_file =  open('targets.csv', 'w')
adv_file = open('advs.csv','w')
pre_dist_file = open('pre_dist.csv','w')
post_dist_file = open('post_dist.csv','w')
pre_prob_file = open('pre_prob.csv','w')
sucess_file = open('success_summary.csv','w')
all_files = [target_file,adv_file,pre_dist_file,post_dist_file,pre_prob_file,sucess_file]
target_writer = csv.writer(target_file, delimiter=',')
adv_writer = csv.writer(adv_file, delimiter=',')
pre_dist_writer = csv.writer(pre_dist_file, delimiter=',')
post_dist_writer = csv.writer(post_dist_file, delimiter=',')
pre_prob_writer = csv.writer(pre_prob_file, delimiter=',')
success_writer = csv.writer(sucess_file, delimiter=',')


#####################################################################################
#						getting outlier ids from test data and poison instances
#####################################################################################
#if the attack is multibase, we find Max_Number_of_poisons to be used as poison bases from the test data
#if allBasesSame = True (default is False), we fix the random seed and therefore the same poison instances would be used for all of the attacks on all targets
#if randomTargets = True (default is False), we do not pick outliers, we just pick the targets to be random (currently fixed seed) among the test instances that get correctly classified
if MultiBase:
	outlier_ids,poisson_ids,_ = find_outliers_from_class_random(n_outlier=n_outlier,out_cls=targClass,base_cls=baseClass,num_poisons=Max_Number_of_poisons, allBasesSame=True, randomTargets= randomTargets)
else: #if the base is going to be just one test example and all poisons are built using the same base, we might as well select that base to be the one with the closest distance
	outlier_ids, poisson_ids,_ = find_outliers_from_class(n_outlier=n_outlier, out_cls=targClass, base_cls=baseClass)

# outlier_ids = outlier_ids[27:28]
# poisson_ids = poisson_ids[27:28]

#####################################################################################
#						preparing the bases
#####################################################################################
def update_X_Base(poisonNumber,new_X_Base,previous_X_Base = None):
	res = previous_X_Base
	if poisonNumber == 0:
		res = np.copy(new_X_Base)
	else:
		res = np.vstack((res, new_X_Base))
	return res

def make_a_poison(driName_tarID_opac_numPoison,poisonNumber,targetNumber,X_target,opacity,IDs_of_Bases_in_test,All_X_Base=None,all_poisons = None,MultiBase=True):
	addNoise = False #initialize adding noise to false -> this is the setting when we have multiple poisons
	if not MultiBase:
		addNoise = True # if the base is the same, we would like to add noise to feature representation of the target

	# get the poisonID which is the id for the base poison instance within the test set
	if MultiBase:
		poisonID = IDs_of_Bases_in_test[poisonNumber]
	else:
		poisonID = IDs_of_Bases_in_test

	# retreive the x and y for the base image
	X_Base_Major = X_test[poisonID].reshape(1,32,32,3)
	Y_Base = Y_test[poisonID]

	#apply the target watermark to the poison base
	coef = opacity/100.
	X_Base = (1-coef)* X_Base_Major + coef*X_target
	#add the new X_Base to the numpy array storing all of the X_bases
	All_X_Base = update_X_Base(poisonNumber=poisonNumber, new_X_Base=X_Base, previous_X_Base=All_X_Base)

	sess = tf.Session()
	print("making poison %d for image %d with opacity %d"%(poisonNumber, targetNumber, int(opacity)))
	apoison = make_instance(i_ter = driName_tarID_opac_numPoison, sess=sess, target=X_target, baseImage=X_Base, baseMajorInpImage=X_Base_Major, addNoise = addNoise)
	sess.close()
	
	
	#save the poison and append it to the list of available poisons
	all_poisons = update_X_Base(poisonNumber=poisonNumber, new_X_Base=apoison, previous_X_Base=all_poisons)
	
	


	return apoison, all_poisons, All_X_Base, Y_Base



#####################################################################################
#						the main things happenning here
#####################################################################################


for i,ID_of_target_in_test in enumerate(outlier_ids):
	for opac in opacs:

		#some initializations
		poisons_loc = poisons[:] #creating a copy of the poisons
		numBuiltPoisons = 0 #initialize the number of poisons built
		IDs_of_Bases_in_test = poisson_ids[i] #take the poison ids for this particular target
		all_poisons = None 
		All_X_Base = None


		# store some statistics
		targets = [opac,ID_of_target_in_test]
		pre_dists = [opac,ID_of_target_in_test]
		post_dists = [opac,ID_of_target_in_test]
		pre_probs = [opac,ID_of_target_in_test]
		advs = [opac,IDs_of_Bases_in_test[numBuiltPoisons]]
		succss = [opac,ID_of_target_in_test]

		# the target X and Y
		X_target = X_test[ID_of_target_in_test].reshape(1,32,32,3)
		Y_target = Y_test[ID_of_target_in_test]

		#numPoisonsExp is the number of experiment by setting the number of poisons
		numPoisonsExp = min(poisons_loc)


		while numBuiltPoisons < Max_Number_of_poisons: #while the number of poison built is less than the maximum that should be built, construct a poison
			# rest the tf graph to minimize memory use
			tf.reset_default_graph()
			apoison, all_poisons, All_X_Base, Y_Base = make_a_poison(driName_tarID_opac_numPoison = "%d_%d_%d"%(ID_of_target_in_test,opac,numPoisonsExp),poisonNumber=numBuiltPoisons,targetNumber=ID_of_target_in_test,X_target=X_target,opacity=opac,IDs_of_Bases_in_test=IDs_of_Bases_in_test,All_X_Base=All_X_Base,all_poisons = all_poisons,MultiBase=MultiBase)
			numBuiltPoisons += 1

			if numBuiltPoisons == numPoisonsExp: #update the experiment when enough poisons have been made after running the experiment
				# run the experiment and train using the poisons
				Y_Base = np.array([Y_Base]*numPoisonsExp).astype(np.int32) #first make all of the Y_Bases

				# do training
				sess = tf.Session()
				adv_cls,target_cls,pre_dist,post_dist,pre_prob = do_training_and_saveWeights("%d_%d_%d"%(ID_of_target_in_test,opac,numPoisonsExp),sess=sess, clean_base_image= All_X_Base,new_poison_sample=all_poisons, y_of_poisonSample =Y_Base ,X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, target=X_target,save_weights = False)
				sess.close()
				tf.reset_default_graph() #free memory be resetting graph

				#checking whether attack was successful or not:
				cur_targ_class = np.argmax(target_cls)
				succss_attack = .0005 #.0005 means that it was misclassified but not to the base class
				statusSuccess = ':) misclassified to some other class'
				if cur_targ_class == baseClass:
					succss_attack = 1 #the attack was successfule
					statusSuccess = ':D misclassified to base class'
				elif cur_targ_class == targClass:
					succss_attack = 0 # the attack was a huge failure
					statusSuccess = ':( NOT misclassified)'
				print("######################################## %d_%d_%d: %s #######################################"%(ID_of_target_in_test,opac,numPoisonsExp,statusSuccess))


				#write summary reports
				targets.append(target_cls)
				advs.append(adv_cls)
				pre_dists.append(pre_dist)
				post_dists.append(post_dist)
				pre_probs.append(pre_prob)
				succss.append(succss_attack)

				#update the experiment
				poisons_loc.pop(poisons_loc.index(numPoisonsExp)) #remove the minimum number of experiments
				if len(poisons_loc) > 0:
					numPoisonsExp = min(poisons_loc)


		target_writer.writerow(targets)
		adv_writer.writerow(advs)
		pre_dist_writer.writerow(pre_dists)
		post_dist_writer.writerow(post_dists)
		pre_prob_writer.writerow(pre_probs)
		success_writer.writerow(succss)
		for file in all_files:
			file.flush()
for file in all_files:
	file.close()



	
			

# # for every target, make




# for i,(ID_of_target_in_test,IDs_of_Bases_in_test) in enumerate(zip(outlier_ids,poisson_ids)):
# 	targets = [i]
# 	advs = [i]
# 	pre_dists = [i]
# 	post_dists = [i]
# 	pre_probs = [i]
# 	for opacity in opacs:
# 		print('Opacity: {}, Iter {}/{}'.format(opacity,i,n_outlier))

# 		X_target_C = X_test[ID_of_target_in_test].reshape(1,32,32,3)
# 		X_target = np.copy(X_target_C)
# 		Y_target = Y_test[ID_of_target_in_test]
# 		# All_X_Base = []
		
# 		if MultiBase:
# 			for poisonID, ID_of_Base_in_test in enumerate(IDs_of_Bases_in_test):
# 				X_Base_C = X_test[ID_of_Base_in_test].reshape(1,32,32,3)
# 				Y_Base = Y_test[ID_of_Base_in_test]
# 				coef = opacity/100.
# 				X_Base_Major = np.copy(X_Base_C)
# 				X_Base = (1-coef)* X_Base_C + coef*X_target
# 				if poisonID == 0:
# 					All_X_Base = np.copy(X_Base)
# 				else:
# 					All_X_Base = np.vstack((All_X_Base, X_Base))
# 				#add some noise to the tagert image if its not the last instance - we want the last instance to be the target so that the distances are accurate
# 				sess = tf.Session()
# 				print("making poison %d for image %d with opacity %d"%(poisonID, i, int(opacity)))
# 				if poisonID != Number_of_poisons - 1:
# 					apoison = make_instance(i_ter = str(i)+'_'+str(opacity),sess=sess, target=X_target, baseImage=X_Base, baseMajorInpImage=X_Base_Major, addNoise = False)
# 					# X_target = X_target_C + np.random.normal(0.,2.5,size=X_target.shape)
# 					# X_target = np.round(X_target)
# 					# X_target = np.clip(X_target,X_target_C.min(),X_target_C.max())
# 				else:
# 					apoison = make_instance(i_ter = str(i)+'_'+str(opacity),sess=sess, target=X_target, baseImage=X_Base, baseMajorInpImage=X_Base_Major, addNoise = False)
# 					# X_target = X_target_C
# 				# plt.show(plt.imshow(X_target.reshape(32,32,3)))
# 				# print(X_target)
# 				sess.close()
# 				tf.reset_default_graph()
# 				if poisonID == 0:
# 					all_poisons = apoison
# 				else:
# 					all_poisons = np.vstack((all_poisons, apoison))
# 			Y_Base = np.array([Y_Base]*Number_of_poisons).astype(np.int32)
# 			# do training
# 			sess = tf.Session()
# 			adv_cls,target_cls,pre_dist,post_dist,pre_prob = do_training_and_saveWeights(str(int(opacity))+'_'+str(Number_of_poisons),sess=sess, clean_base_image= All_X_Base,new_poison_sample=all_poisons, y_of_poisonSample =Y_Base ,X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, target=X_target_C,save_weights = False)
# 			sess.close()
# 		else:
# 			#we would like to start from the same base instance (one that is closest to the target image in feature space) and just make multiple poisons by adding a small random perturbation to the target instance feature representation
# 			for poisonID in range(Number_of_poisons):
# 				X_Base_C = X_test[IDs_of_Bases_in_test].reshape(1,32,32,3)
# 				Y_Base = Y_test[IDs_of_Bases_in_test]
# 				coef = opacity/100.
# 				X_Base_Major = np.copy(X_Base_C)
# 				X_Base = (1-coef)* X_Base_C + coef*X_target
# 				if poisonID == 0:
# 					All_X_Base = np.copy(X_Base)
# 				else:
# 					All_X_Base = np.vstack((All_X_Base, X_Base))
# 				#add some noise to the tagert image if its not the last instance - we want the last instance to be the target so that the distances are accurate
# 				sess = tf.Session()
# 				print("making poison %d for image %d with opacity %d"%(poisonID, i, int(opacity)))
# 				if poisonID != Number_of_poisons - 1:
# 					apoison = make_instance(i_ter = str(i)+'_'+str(opacity),sess=sess, target=X_target, baseImage=X_Base, baseMajorInpImage=X_Base_Major, addNoise = True)
# 					# X_target = X_target_C + np.random.normal(0.,2.5,size=X_target.shape)
# 					# X_target = np.round(X_target)
# 					# X_target = np.clip(X_target,X_target_C.min(),X_target_C.max())
# 				else:
# 					apoison = make_instance(i_ter = str(i)+'_'+str(opacity),sess=sess, target=X_target, baseImage=X_Base, baseMajorInpImage=X_Base_Major, addNoise = False)
# 					# X_target = X_target_C
# 				# plt.show(plt.imshow(X_target.reshape(32,32,3)))
# 				# print(X_target)
# 				sess.close()
# 				tf.reset_default_graph()
# 				if poisonID == 0:
# 					all_poisons = apoison
# 				else:
# 					all_poisons = np.vstack((all_poisons, apoison))
# 			Y_Base = np.array([Y_Base]*Number_of_poisons).astype(np.int32)
# 			# do training
# 			sess = tf.Session()
# 			adv_cls,target_cls,pre_dist,post_dist,pre_prob = do_training_and_saveWeights(str(int(opacity))+'_'+str(Number_of_poisons),sess=sess, clean_base_image= All_X_Base,new_poison_sample=all_poisons, y_of_poisonSample =Y_Base ,X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, target=X_target_C,save_weights = False)
# 			sess.close()

# 		tf.reset_default_graph()
# 		targets.append(target_cls)
# 		advs.append(adv_cls)
# 		pre_dists.append(pre_dist)
# 		post_dists.append(post_dist)
# 		pre_probs.append(pre_prob)

# 	target_writer.writerow(targets)
# 	adv_writer.writerow(advs)
# 	pre_dist_writer.writerow(pre_dists)
# 	post_dist_writer.writerow(post_dists)
# 	pre_prob_writer.writerow(pre_probs)
# 	for file in all_files:
# 		file.flush()
# for file in all_files:
# 	file.close()



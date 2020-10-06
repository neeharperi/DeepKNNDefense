"""
utilities needed for the iterative poisonong
"""
import tensorflow as tf
import re
import numpy as np
import math
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import time
from scipy import misc
from copy import deepcopy
import os
# import imageio


########################################################################################################################
####                                            Parameters
########################################################################################################################

dir_params = './PreTrainedNetParams/'

#for retraining the CNN
batch_size = 128
Nepochs = 10

MaxIter = 12000 #maximum iterations for the optimziation process
# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
IMAGE_SIZE = 32

########################################################################################################################
####                                            tf graphs
########################################################################################################################



def make_Cnet_andGetTensorsFor_poisonMaking(learning_rate, coeff_TV):
    """making conv net for generating poisons for CIFAR-10"""
    x = tf.Variable(np.zeros((1,32,32,3)).astype(np.float32)) # the image that we are trying to make
    ft = tf.placeholder(tf.float32, shape=(None, 192)) #placeholder for feature representation of the target -> what we are trying to get close to
    bi = tf.placeholder(tf.float32, shape=(None,32,32,3)) # placeholder for base input image representation (pixel values)
    y = tf.placeholder(tf.int32, shape=(None, 10)) #placeholder for label
    learning_rateTF = tf.placeholder(tf.float32, shape=[])
    # some operations
    modify_op = x.assign(bi) #assigns an initial value to the variable
    clip_op = tf.clip_by_value(x, 0.0, 1.0)
    rel_change = tf.norm(x-bi)/tf.norm(x)

    #conv1
    kernel1 = tf.constant(np.load(dir_params+'kernel1.npy'))
    conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')
    bias1_conv = tf.constant(np.load(dir_params+'bias1_conv.npy'))
    pre_activation1 = tf.nn.bias_add(conv1, bias1_conv)
    conv1 = tf.nn.relu(pre_activation1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    #conv2
    kernel2 = tf.constant(np.load(dir_params+'kernel2.npy'))
    conv2 = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')
    bias2_conv = tf.constant(np.load(dir_params+'bias2_conv.npy'))
    pre_activation2 = tf.nn.bias_add(conv2, bias2_conv)
    conv2 = tf.nn.relu(pre_activation2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = flatten(pool2)#tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights1 = tf.constant(np.load(dir_params+'weights1.npy'))
    bias_w1 = tf.constant(np.load(dir_params+'bias_w1.npy'))
    local3 = tf.nn.relu(tf.matmul(reshape, weights1) + bias_w1)

    # local4
    weights2 = tf.constant(np.load(dir_params+'weights2.npy'))
    bias_w2 = tf.constant(np.load(dir_params+'bias_w2.npy'))
    local4 = tf.nn.relu(tf.matmul(local3, weights2) + bias_w2)

    featsTensor = local4
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    fc_w = tf.constant(np.load(dir_params+'fc_w.npy'))
    fc_b = tf.constant(np.load(dir_params+'fc_b.npy'))
    logits = tf.add(tf.matmul(local4, fc_w), fc_b)

    softout = tf.nn.softmax(logits)
    pred = tf.argmax(softout,1)
    accurs = tf.reduce_mean( tf.cast(tf.equal(pred,tf.cast(y,tf.int64)),tf.float32))

    ###################### things needed for training and making the poison
    ############ note that featsTensor is the feature representation
    the_diff = tf.norm(tf.subtract(featsTensor,ft))
    loss_poison = tf.add(tf.norm(tf.subtract(featsTensor,ft)), tf.multiply(coeff_TV,tf.image.total_variation(x)[0]))
    train_step_poison = tf.train.AdamOptimizer(learning_rateTF).minimize(loss_poison)

    return x, ft, bi, modify_op, clip_op, train_step_poison, loss_poison, the_diff, softout, pred, accurs, featsTensor, learning_rateTF, rel_change


def make_Cnet_andGetTensorsFor_training(learning_rate):
    """making convnet and loading the pretrained weights for training"""
    #image placeholder
    x = tf.placeholder(tf.float32, shape=[None,IMAGE_SIZE,IMAGE_SIZE,3])
    y = tf.placeholder(tf.int32, shape=[None])

    #conv1
    kernel1 = tf.Variable(np.load(dir_params+'kernel1.npy'))
    conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')
    bias1_conv = tf.Variable(np.load(dir_params+'bias1_conv.npy'))
    pre_activation1 = tf.nn.bias_add(conv1, bias1_conv)
    conv1 = tf.nn.relu(pre_activation1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    #conv2
    kernel2 = tf.Variable(np.load(dir_params+'kernel2.npy'))
    conv2 = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')
    bias2_conv = tf.Variable(np.load(dir_params+'bias2_conv.npy'))
    pre_activation2 = tf.nn.bias_add(conv2, bias2_conv)
    conv2 = tf.nn.relu(pre_activation2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = flatten(pool2)#tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights1 = tf.Variable(np.load(dir_params+'weights1.npy'))
    bias_w1 = tf.Variable(np.load(dir_params+'bias_w1.npy'))
    local3 = tf.nn.relu(tf.matmul(reshape, weights1) + bias_w1)

    # local4
    weights2 = tf.Variable(np.load(dir_params+'weights2.npy'))
    bias_w2 = tf.Variable(np.load(dir_params+'bias_w2.npy'))
    local4 = tf.nn.relu(tf.matmul(local3, weights2) + bias_w2)

    featsTensor = local4
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    fc_w = tf.Variable(np.load(dir_params+'fc_w.npy'))
    fc_b = tf.Variable(np.load(dir_params+'fc_b.npy'))
    logits = tf.add(tf.matmul(local4, fc_w), fc_b)

    varNames = ['kernel1', 'bias1_conv', 'kernel2', 'bias2_conv', 'weights1', 'bias_w1', 'weights2', 'bias_w2', 'fc_w', 'fc_b']


    softout = tf.nn.softmax(logits)
    pred = tf.argmax(softout,1)
    accurs = tf.reduce_mean( tf.cast(tf.equal(pred,tf.cast(y,tf.int64)),tf.float32))

    loss = tf.add(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) , tf.add(tf.multiply(tf.nn.l2_loss(weights1),0.004) , tf.multiply(tf.nn.l2_loss(weights2),0.004)))

    # Decay the learning rate exponentially based on the number of steps.
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    allVars = []
    varNames = []
    return x, y, train_step, softout, pred, accurs, featsTensor, allVars, varNames, loss, fc_w, fc_b
	

def adjust_moving_back(baseMajorInpImage,baseInpImage,currentImage,coeff_sim_inp,learning_rate,eps=0.1):
	# return baseMajorInpImage + np.maximum(np.minimum(currentImage - baseMajorInpImage,eps) ,-eps) #np.clip(currentImage)

	###  Prox operator when using an l2 penalty
	base_coef = coeff_sim_inp#/4.0 	# coefficient of the base image term
	major_coef = 0.*coeff_sim_inp		# coefficient on major image term
	# The prox operator for this two-term objective function is simply a weighted average of the base, major, and current image
	return (base_coef*learning_rate*baseInpImage + major_coef*learning_rate*baseMajorInpImage + currentImage) / (base_coef*learning_rate + major_coef*learning_rate + 1)

	# return currentImage + learning_rate/2.*coeff_sim_inp*(baseInpImage - currentImage) + learning_rate*coeff_sim_inp*(baseMajorInpImage - currentImage)

def assign_var_image(sess, baseInpImage,modifyInitialVarOp, baseTensorPlaceHolder):
	#re assign the variable to the be the base again
	sess.run(modifyInitialVarOp, feed_dict={baseTensorPlaceHolder: baseInpImage})
	return

def saveEm(iterNumber, optIter,numpyImage,the_diffHere, directory=""):

	if not os.path.exists(directory):
		os.makedirs(directory)
	name=str(iterNumber)+'-'+str(optIter)+'-'+str(the_diffHere)+'.png'
	#plt.imsave(directory + name, numpyImage.reshape(32,32,3))
	misc.imsave(directory+name, numpyImage.reshape(32,32,3))
	# imageio.imwrite(directory+name, numpyImage.reshape(32,32,3))
	np.save(directory + name[:-5]+'.npy',numpyImage)
	return



def do_the_optimization(relchange_op,imgNum, sess, feed_dict, baseMajorInpImage,baseInpImage,imageinpvar, clip_op,modify_op,bi,the_diffTensor, training_op, loss_tensor, softoutTensor,coeff_sim_inp,learning_rate,learning_rateTF, MaxIter=MaxIter,EveryThisNThen=500, tol=1e-4):
	decayCoef = 0.2
	stopping_tol = 1e-5
	old_image = sess.run(imageinpvar)
	for iter in range(MaxIter):

		old_obj = sess.run(loss_tensor, feed_dict=feed_dict)
		#save images every now and then
		if iter % EveryThisNThen == 0:
			# print(imgNum,':', iter,'-',np.argmax(sess.run(softoutTensor)),sess.run(softoutTensor)," - obj func:", old_obj, 'diff:', sess.run(the_diffTensor, feed_dict=feed_dict))
			the_diffHere = sess.run(the_diffTensor, feed_dict=feed_dict)  	#get the diff
			theNPimg = sess.run(imageinpvar)  								#get the image
			# saveEm(iterNumber=imgNum,optIter=iter, numpyImage=theNPimg, the_diffHere=the_diffHere, directory='interim_images/'+str(imgNum)+'/')
		# rel_change_val = sess.run(relchange_op, feed_dict={bi:old_image})#tf.norm(imageinpvar-old_image)/tf.norm(imageinpvar), feed_dict=feed_dict_lr)
					#saveEm(iterNumber="IN"+str(imgNum)+'-'+str(i), numpyImage=theNPimg, the_diffHere = the_diffHere, directory="/Users/ashafahi/Documents/UMD-New/Spring2018/tomg/poison2/outlier/Interim/")
		#print the reports
		# print(imgNum,':', i,'-',ThreeN8[np.argmax(sess.run(softoutTensor))],sess.run(softoutTensor)," - obj func:", myObjF, 'diff:', sess.run(the_diffTensor, feed_dict=feed_dict))

		# gradient update
		feed_dict_lr = {learning_rateTF:learning_rate}
		feed_dict_lr.update(feed_dict)
		sess.run(training_op, feed_dict=feed_dict_lr)

		# check stopping condition:  compute relative change in image between iterations
		rel_change_val = sess.run(relchange_op, feed_dict={bi:old_image})#tf.norm(imageinpvar-old_image)/tf.norm(imageinpvar), feed_dict=feed_dict_lr)
		# print " (%d) Rel change =  %0.5e   |   lr = %0.5e |   obj = %0.10e"%(iter,rel_change_val,learning_rate,old_obj)

		# print " (%d) Rel change =  %0.5e   |   lr = %0.5e |   obj = %0.10e"%(iter,rel_change_val,learning_rate,old_obj)
		if rel_change_val<stopping_tol:
			break

		# compute new objective value
		new_obj = sess.run(loss_tensor, feed_dict=feed_dict)

		# The backward step in the forward-backward iteration
		new_candid = sess.run(imageinpvar)
		new_candid = adjust_moving_back(baseMajorInpImage=baseMajorInpImage,baseInpImage=baseInpImage ,currentImage = new_candid, coeff_sim_inp=coeff_sim_inp, learning_rate=learning_rate)
		#assign_var_image(sess=sess,baseInpImage=new_candid, modifyInitialVarOp=modify_op, baseTensorPlaceHolder=bi)
		

		# If the objective went up, then learning rate is too big.  Chop it, and throw out the latest iteration
		if  new_obj >= old_obj:
			learning_rate *= decayCoef
			# sess.run(imageinpvar.assign(old_image))
			assign_var_image(sess=sess,baseInpImage=old_image, modifyInitialVarOp=modify_op, baseTensorPlaceHolder=bi)
		else:
			old_image = new_candid
			# sess.run(imageinpvar.assign(new_candid))
			assign_var_image(sess=sess,baseInpImage=new_candid, modifyInitialVarOp=modify_op, baseTensorPlaceHolder=bi)


	#save final images
	saveEm(iterNumber=imgNum,optIter='final', numpyImage=sess.run(imageinpvar), the_diffHere=sess.run(the_diffTensor, feed_dict=feed_dict), directory='debug_images/'+str(imgNum)+'/')
	return sess.run(imageinpvar)

def batch_indices(batchID,NdataTrain,batch_size):
	start = int(batchID*batch_size)
	end = int((batchID+1)*batch_size)
	if end > NdataTrain:
		shift = end - NdataTrain
		start -= shift
		end -= shift
	return start, end


def get_prediction_and_show(sample,sess,softout, pred,x,showImage = False, showClassPred=False):

	if showClassPred:
		print("class prediction:", np.argmax(sess.run(pred, feed_dict={x:sample})))
	cls_probs = sess.run(softout, feed_dict={x:sample})
	print("class probs:", cls_probs)
	if showImage:
		plt.imshow(sample.reshape(32,32,3))
		plt.show()
	return cls_probs


def get_cosine_weights(weights1, weights2, this_class):
	weights1_c = weights1[:,this_class]
	weights2_c = weights2[:, this_class]
	c = np.dot(weights2_c,weights1_c)/np.linalg.norm(weights2_c)/np.linalg.norm(weights1_c)
	return c


def batch_indices_forFeats(batchID,NdataTrain,batch_size):
	start = int(batchID*batch_size)
	end = int((batchID+1)*batch_size)
	if end > NdataTrain:
		end = NdataTrain
	return start, end


def get_train_feats(X_train, sess, x, featTensor, batch_size):
	all_feat_reps  = []
	nb_batches = int(math.ceil(float(len(X_train))/batch_size))
	for batch in range(nb_batches):
		start, end = batch_indices_forFeats(batch, len(X_train), batch_size)
		these_feats = sess.run(featTensor, feed_dict = {x:X_train[start:end]})
		if batch == 0:
			all_feat_reps = these_feats
		else:
			all_feat_reps = np.vstack((all_feat_reps, these_feats))
	return all_feat_reps


def handle_all_poisons(whichTask,allPoisons):
	results4all = []
	for i in len(allPoisons):
		print "hi"

	return results4all

def get_accuracy_test(sess, x,y,accurs, X_test, Y_test, batch_size=128):
	res  = 0.
	nb_batches = int(math.ceil(float(len(X_test))/batch_size))
	for batch in range(nb_batches):
		start, end = batch_indices_forFeats(batch, len(X_test), batch_size)
		these_total_accurs = sess.run(accurs, feed_dict = {x:X_test[start:end], y:Y_test[start:end]})*(end-start)
		if batch == 0:
			res = these_total_accurs
		else:
			res += these_total_accurs
	print("total accuracy on test is:%.5e"%(res/len(X_test)))
	return 

def perform_one_entire_epoch(opacAndNumPoison,loss_WRNet,X_train, Y_train, X_test, Y_test, accurs, train_step, sess, batch_size, y, x,featTensor, clean_base_image, allPoisons, target, fc_w, fc_b, save_feat_reps):
	get_accuracy_test(sess, x,y,accurs, X_test, Y_test, batch_size=batch_size)
	feed_dict_for_clean_image_base = {x:clean_base_image}
	feed_dict_for_adv_image = {x:allPoisons}
	feed_dict_for_target = {x:target}
	#getting feat reps before any training on the poison
	all_feat_reps = get_train_feats(X_train=X_train, sess=sess, x=x, featTensor=featTensor, batch_size=batch_size)
	#now saving them based on epoch
	directory = str(opacAndNumPoison)+'/'
	if not os.path.exists(directory):
		os.makedirs(directory)

	if save_feat_reps:
		np.save(directory+"%s-X_tr_feats-beforePoison.npy"%opacAndNumPoison,all_feat_reps)
		np.save(directory+"%s-tar_feats-beforePoison.npy"%opacAndNumPoison, sess.run(featTensor, feed_dict=feed_dict_for_target))
	for i,bs in enumerate(clean_base_image):
		this_instance = sess.run(featTensor, feed_dict={x:bs.reshape(1,32,32,3)})
		if i == 0:
			allCleanInstanceBases = this_instance
		else:
			allCleanInstanceBases = np.vstack((allCleanInstanceBases,this_instance))

	if save_feat_reps:
		np.save(directory+"%s-base_feats-beforePoison.npy"%opacAndNumPoison,allCleanInstanceBases)#sess.run(featTensor, feed_dict=feed_dict_for_clean_image_base))
		np.save(directory+"%s-Y_tr_beforePoison.npy"%opacAndNumPoison, Y_train)

	global Nepochs
	for nep in range(Nepochs):
		total_loss = 0.
		totalaccurs = 0.
		st_time = time.time()
		rng = np.random.RandomState()
	
		nb_batches = int(math.ceil(float(len(X_train))/batch_size))
		index_shuf = list(range(len(X_train)))
		rng.shuffle(index_shuf)
		for batch in range(nb_batches):
			# Compute batch start and end indices
			start, end = batch_indices(batch, len(X_train), batch_size)
			# Perform one training step
			feed_dict = {x: X_train[index_shuf[start:end]], y: Y_train[index_shuf[start:end]]}
			sess.run(train_step,feed_dict=feed_dict)
			loss_miniBatch = sess.run(loss_WRNet,feed_dict=feed_dict).sum()
			train_accurs_miniBatch = sess.run(accurs, feed_dict=feed_dict)*(end-start)
			total_loss += loss_miniBatch
			totalaccurs += train_accurs_miniBatch
			# if batch == nb_batches - 1:
			# 	loss_miniBatch = sess.run(loss_WRNet,feed_dict=feed_dict).sum()
			# 	total_loss += loss_miniBatch
			# 	print("loss: %.7e"%total_loss)
			# print("accuracy on this minibatch (%d-%d/%d) is: %.5e and loss is %.5e "%(nep,batch,nb_batches,sess.run(accurs,feed_dict=feed_dict),loss_miniBatch))
		assert end >= len(X_train)  # Check that all examples were used
		print "total accuracy on train is:", totalaccurs/(nb_batches*batch_size)
		get_accuracy_test(sess, x,y,accurs, X_test, Y_test, batch_size=128)
		dist_to_target = np.linalg.norm(sess.run(featTensor, feed_dict={x:allPoisons[0].reshape(1,32,32,3)}) - sess.run(featTensor, feed_dict=feed_dict_for_target))
		dist_to_base = np.linalg.norm(sess.run(featTensor, feed_dict={x:allPoisons[0].reshape(1,32,32,3)}) - sess.run(featTensor, feed_dict=feed_dict_for_clean_image_base))
		fc_w_curr = sess.run(fc_w)
		fc_b_curr = sess.run(fc_b)
		fc_w_old = np.load(dir_params+'fc_w.npy')
		if save_feat_reps:
			np.save(directory+'%s-fc_w-beforePoison.npy'%opacAndNumPoison,fc_w_old)
		fc_b_old = np.load(dir_params+'fc_b.npy')
		if save_feat_reps:
			np.save(directory+'%sfc_b-beforePoison.npy'%opacAndNumPoison, fc_b_old)
		norm_of_change_in_weights = np.linalg.norm(fc_w_curr - fc_w_old)
		norm_of_change_in_bias = np.linalg.norm(fc_b_curr - fc_b_old)

		cosine_base = get_cosine_weights(fc_w_old, fc_w_curr, 6)
		cosine_targ = get_cosine_weights(fc_w_old, fc_w_curr, 0)
		print("******************************>>>>>>>>>>>>>")
		print("BASE** cosine between before training and training: %.5e - diff in bias: %.5e"%(cosine_base,fc_b_old[6]-fc_b_curr[6]))
		print("Target** cosine between before training and training: %.5e - diff in bias: %.5e"%(cosine_targ,fc_b_old[0]-fc_b_curr[0]))
		print("******************************<<<<<<<<<<<")
		print("epoch: %d is done in %.5e minutes - total loss: %.5e - dist from target:%.5e - dist of adv form clean image:%.5e"%(nep,(time.time()-st_time)/60.0,total_loss,dist_to_target, dist_to_base))
		if save_feat_reps:
			# now getting the training feats
			all_feat_reps = get_train_feats(X_train=X_train, sess=sess, x=x, featTensor=featTensor, batch_size=batch_size)
			#now saving them based on epoch
			np.save(directory+"%s-X_tr_feats-%d.npy"%(opacAndNumPoison,nep),all_feat_reps)
			np.save(directory+"%s-fc_w-%d.npy"%(opacAndNumPoison,nep),fc_w_curr)
			np.save(directory+"%s-fc_b-%d.npy"%(opacAndNumPoison,nep),fc_b_curr)
			np.save(directory+"%s-tar_feats-%d.npy"%(opacAndNumPoison,nep), sess.run(featTensor, feed_dict=feed_dict_for_target))
			for i,bs in enumerate(clean_base_image):
				this_instance = sess.run(featTensor, feed_dict={x:bs.reshape(1,32,32,3)})
				if i == 0:
					allCleanInstanceBases = this_instance
				else:
					allCleanInstanceBases = np.vstack((allCleanInstanceBases,this_instance))

			np.save(directory+"%s-base_feats-%d.npy"%(opacAndNumPoison,nep),allCleanInstanceBases) #sess.run(featTensor, feed_dict=feed_dict_for_clean_image_base))
		# acc_now = sess.run(accurs, feed_dict={x:X_test , y:Y_test})
		# print("its accuracy on test was: " + str(acc_now) + " train acc:" + str(sess.run(accurs, feed_dict={x:X_train,y:Y_train})))

def show_res_summary(opacAndNumPoison,loss_WRNet,clean_base_image,IterNum, allPoisons,y_of_poisonSample,clean_img, X_train, Y_train, X_test, Y_test, featTensor, softout, pred, accurs,train_step, batch_size ,sess, classTensor, inputImageTensor, fc_w, fc_b, save_feat_reps):

	
	#new image
	new_adv_img = allPoisons[0].reshape(1,32,32,3)
	# it's class
	thisIsTheClass = np.argmax(sess.run(softout, feed_dict={inputImageTensor:new_adv_img}))
	new_adv_y = y_of_poisonSample 
	print("\n**********************")
	print("<<<<<< before training >>>>>>>")
	# print("iteration %s, class of first adv: %s but its true class is: %d" %(IterNum, set_of_those[thisIsTheClass],new_adv_y))
	all_pre_dist = []
	for this_adv in range(len(allPoisons)):
		pre_dist = np.linalg.norm(sess.run(featTensor, feed_dict={inputImageTensor:allPoisons[this_adv].reshape(1,32,32,3)}) - sess.run(featTensor, feed_dict={inputImageTensor:clean_img}))
		all_pre_dist.append(pre_dist)
		print("distance of poison %d  from target in feat space: %.5e" %(this_adv,pre_dist))
	print("adv probs:")
	for this_adv in range(len(allPoisons)):
		adv_prbs = sess.run(softout, feed_dict={inputImageTensor:allPoisons[this_adv].reshape(1,32,32,3)})
		print("poison %d's probabality class:"%this_adv,adv_prbs)
	# do training on one sample
	# perform_one_training_iteration(x_sample=new_adv_img,y_sample=new_adv_y)
	X_train = np.vstack((X_train, allPoisons))# new_adv_img))
	Y_train = np.append(Y_train, y_of_poisonSample)
	print("target probs:")
	pre_prob = get_prediction_and_show(sample=clean_img, sess=sess, softout=softout, pred=pred,  x= inputImageTensor)
	perform_one_entire_epoch(opacAndNumPoison,loss_WRNet,X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,accurs=accurs,batch_size=batch_size,train_step=train_step,sess=sess, y=classTensor, x=inputImageTensor, featTensor=featTensor,clean_base_image=clean_base_image, allPoisons=allPoisons, target=clean_img, fc_w=fc_w, fc_b=fc_b, save_feat_reps=save_feat_reps)
	#now get probs of clean
	print("<<<<<< after training >>>>>>>")
	all_post_dist = []
	for this_adv in range(len(allPoisons)):
		post_dist = np.linalg.norm(sess.run(featTensor, feed_dict={inputImageTensor:allPoisons[this_adv].reshape(1,32,32,3)}) - sess.run(featTensor, feed_dict={inputImageTensor:clean_img}))
		all_post_dist.append(post_dist)
		print("distance of poison %d  from target in feat space: %.5e" %(this_adv,post_dist))
	print("adv probs:")
	all_adv_cls = []
	for this_adv in range(len(allPoisons)):
		adv_cls = get_prediction_and_show(sample=allPoisons[this_adv].reshape(1,32,32,3), sess=sess, softout=softout, pred=pred, x = inputImageTensor)
		all_adv_cls.append(adv_cls)
		print(adv_cls)
	if np.argmax(adv_cls) != new_adv_y[0]:
		print("***** not enough training!**** try more epochs or a higher learning rate")
	print("target probs:")
	target_cls = get_prediction_and_show(sample=clean_img, sess=sess, softout=softout, pred=pred, x = inputImageTensor)
	curr_tar_class = np.argmax(target_cls)
	if np.argmax(pre_prob) != curr_tar_class:
		print(" :) target as been knocked out of its class")
		if curr_tar_class == new_adv_y[0]:
			print(" :D the target now has the class of the adversarial")
	return all_adv_cls,target_cls,all_pre_dist,all_post_dist,pre_prob


def transposeFromTorch2TF(v):
    if v.ndim == 4:
        return v.transpose(2,3,1,0)
    elif v.ndim == 2:
        return v.transpose()
    else:
        return v

def take_prams_and_make_tensors(params, variable=False):
	for k,v in sorted(params.items()):
		params[k] = transposeFromTorch2TF(v)
		print(k, params[k].shape)
	if not variable:
		for k,v in params.items():
			params[k] = tf.constant(v,name=k)
	else:
		for k,v in params.items():
			params[k] = tf.Variable(v,name=k)
	return params

def make_instance(i_ter,sess, target, baseImage, baseMajorInpImage, addNoise=False):
	coeff_TV = 0.0
	coeff_sim_inp = .1#.25
	learning_rate = 0.01
	stdNoise = 0.05


	#build tensor graph and get the required tensors
	x, ft, bi, modify_op, clip_op, train_step_poison, loss_poison, the_diff, softout, pred, accurs, fc2, learning_rateTF, rel_change  = make_Cnet_andGetTensorsFor_poisonMaking(learning_rate=learning_rate, coeff_TV=coeff_TV)
	# get the inital values for vars
	sess.run(tf.global_variables_initializer())

	#here's where the poison instance is being made
	new_adv_img = target
	#force input var to be this image to get its feature representation
	assign_var_image(sess=sess, baseInpImage=new_adv_img,modifyInitialVarOp=modify_op, baseTensorPlaceHolder=bi)
	feat_rep_adv = sess.run(fc2)
	if addNoise:
		feat_rep_adv += np.random.normal(0,stdNoise,size=feat_rep_adv.shape)
	#now do optimization starting from random noise but before that initialize the base image to random noise or something you like -> start from image from previous iteration
	assign_var_image(sess=sess, baseInpImage=baseImage, modifyInitialVarOp=modify_op, baseTensorPlaceHolder=bi)
	print('**************************************  image: %s *****************************' %i_ter)
	#save the asversarial image
	saveEm(iterNumber=str(i_ter),optIter='main', numpyImage=new_adv_img, the_diffHere=0, directory='debug_images/'+str(i_ter)+'/')
	prevFinalImage = do_the_optimization(relchange_op=rel_change,imgNum=i_ter, sess=sess, feed_dict={ft:feat_rep_adv, bi:baseImage}, baseMajorInpImage=baseMajorInpImage,baseInpImage=baseImage,imageinpvar=x, clip_op=clip_op,modify_op=modify_op,bi=bi,the_diffTensor=the_diff, training_op=train_step_poison, loss_tensor=loss_poison, softoutTensor=softout, coeff_sim_inp=coeff_sim_inp, learning_rate=learning_rate, learning_rateTF=learning_rateTF)

	return prevFinalImage

def do_training_and_saveWeights(opacAndNumPoison, sess, clean_base_image,new_poison_sample,y_of_poisonSample, X_train, X_test, Y_train, Y_test, target,save_weights = False, save_feat_reps=False):
	learning_rate = 1.8e-5
	global batch_size

	#load tensor graph:
	x, y, train_step, softout, pred, accurs, fc2, allVars, varNames, loss_WRNet, fc_w, fc_b  = make_Cnet_andGetTensorsFor_training(learning_rate)
	#iniitalize all vars
	sess.run(tf.global_variables_initializer())
	adv_cls,target_cls,pre_dist,post_dist,pre_prob = show_res_summary(opacAndNumPoison, loss_WRNet,clean_base_image,IterNum=0, allPoisons=new_poison_sample,y_of_poisonSample=y_of_poisonSample,clean_img=target, X_train=X_train, Y_train=Y_train, X_test=X_test , Y_test=Y_test, featTensor=fc2, softout=softout, pred=pred, accurs=accurs,train_step=train_step,batch_size=batch_size,sess=sess,classTensor=y, inputImageTensor=x, fc_w=fc_w, fc_b=fc_b, save_feat_reps=save_feat_reps)
	if save_weights:
		for vrN,vr in zip(varNames,allVars):
			np.save(vrN+'.npy', sess.run(vr))
	return adv_cls,target_cls,pre_dist,post_dist,pre_prob
	


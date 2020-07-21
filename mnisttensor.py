from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import params
import sys
import numpy as np
import time
from modeltensor import LeNet_5


def train(data):
	x = tf.placeholder(tf.float32, shape= (None, ) + (28, 28, 1))
	y = tf.placeholder(tf.float32, shape= (None, 10))

	lenet5 = LeNet_5(x, y)
	optimizer = tf.train.AdamOptimizer(params.learning_rate).minimize(lenet5.loss)
	saver = tf.train.Saver(tf.trainable_variables())

	list_train_acc = [0]
	list_val_acc = [0]
	list_test_acc = [0]

	with tf.Session() as sess:
		print("Starting Training!")
		sess.run(tf.global_variables_initializer())
		train_num_batch = data.train.num_examples // params.batch_size

		if params.have_validation:
			validation = True
			val_num_batch = data.validation.num_examples // params.batch_size
		else:
			validation = False

		test_num = data.test.num_examples
		test_x = data.test.images
		test_x = np.reshape(test_x, (test_num, 28, 28, 1))
		test_y = data.test.labels

		for epoch in range(params.epochs):
			begin = time.time()
			train_acc = []
			for __ in range(train_num_batch):
				xt, yt = data.train.next_batch(params.batch_size)
				reshape_xt = np.reshape(xt, (params.batch_size, 28, 28, 1))
				feed_dict_train = {
					x: reshape_xt,
					y: yt,
					lenet5.keep_prob: 0.8
				}
				__, acc = sess.run([optimizer, lenet5.accuracy], feed_dict= feed_dict_train)
				train_acc.append(acc)
			train_acc_mean = np.mean(train_acc)
			list_train_acc.append(train_acc_mean)

			feed_dict_test = {
				x: test_x,
				y: test_y,
				lenet5.keep_prob: 1.0
			}

			test_acc = sess.run(lenet5.accuracy, feed_dict= feed_dict_test)
			list_test_acc.append(test_acc)

			if validation:
				validation_acc = []
				for __ in range(val_num_batch):
					xv, yv = data.validation.next_batch(params.batch_size)
					reshape_xv = np.reshape(xv, (params.batch_size, 28, 28, 1))
					feed_dict_val = {
						x: reshape_xv,
						y: yv,
						lenet5.keep_prob: 1.0
					}
					acc = sess.run(lenet5.accuracy, feed_dict= feed_dict_val)
					validation_acc.append(acc)
				validation_acc_mean = np.mean(validation_acc)
				list_val_acc.append(validation_acc_mean)

				print("Epoch: %d, time: %ds, train accuracy: %.4f, test accuracy: %.4f, validation accuracy: %.4f."%(epoch, time.time()-begin, train_acc_mean, test_acc, validation_acc_mean))
			else:
				print("Epoch: %d, time: %ds, train accuracy: %.4f."%(epoch, time.time()-begin, train_acc_mean))
			sys.stdout.flush()

			if (epoch + 1)%10==0:
				save_model = os.path.join(params.model_dir, "Model_Lenet-5.ckpt")
				saver.save(sess, save_model)

			max_val_acc = max(list_val_acc)
			if list_val_acc[-1] < max_val_acc - params.valid_eps:
				break

		save_model = os.path.join(params.model_dir, "Model_Lenet-5.ckpt")
		saver.save(sess, save_model)
		print("Finish training!")

	return 1


if __name__ == '__main__':
	mnist = input_data.read_data_sets("data/", one_hot=True)

	if not os.path.exists(params.model_dir):
		os.makedirs(params.model_dir)
	train(mnist)
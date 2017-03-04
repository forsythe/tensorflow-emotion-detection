import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import cv2

import time

###################
emotion_name = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral", "unknown"]

###################NEURAL NETWORK PROPERTIES

n_examples = 28709
n_classes = 7

capacity = 2000
batch_size = 1000
min_after_dequeue = 1000
hm_epochs = 1

###################TENSORFLOW
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
FLAGS = tf.app.flags.FLAGS

x = tf.placeholder('float', [None, 2304]) #48*48=2304
y = tf.placeholder('float',[None, n_classes])


keep_rate = 0.6
keep_prob = tf.placeholder(tf.float32)

weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
			'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
			'W_fc': tf.Variable(tf.random_normal([12*12*64, 1024])),
			'out': tf.Variable(tf.random_normal([1024, n_classes]))}
biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
			'b_conv2': tf.Variable(tf.random_normal([64])),
			'b_fc': tf.Variable(tf.random_normal([1024])),
			'out': tf.Variable(tf.random_normal([n_classes]))}

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
choice = input("[load], [train] from scratch, or [continue] training? ")
while (not (choice == "train")) and (not (choice == "load") and (not (choice == "continue"))):
	choice = input("Invalid input. Please try again.")

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding='SAME')

def conv_neural_network_model(data):
	x = tf.reshape(data, shape=[-1, 48, 48, 1])
	conv1 = conv2d(x, weights['W_conv1']) + biases['b_conv1']
	conv1 = maxpool2d(conv1)
	conv2 = conv2d(conv1, weights['W_conv2']) + biases['b_conv2']
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2, [-1, 12*12*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)
	output = tf.matmul(fc, weights['out'])+biases['out']

	return output

result = conv_neural_network_model(x)


def val_to_one_hot(x):
	ans = np.array([0, 0, 0, 0, 0, 0, 0])
	ans[x]=1
	return ans

def plot_image(images, emotion_num, prediction, prediction_best_guess):
	images = np.reshape(images, [48, 48])
	correct_emotion = emotion_name[emotion_num]
	best_guess = emotion_name[prediction_best_guess]
	txt = ""
	for k in range(n_classes):
		txt +=  str(emotion_name[k]) + ": " + str(round(prediction[0][k], 3)) + "\n"

	fig = plt.figure()
	left = fig.add_subplot(121)
	title("Correct emotion: " + correct_emotion+"\n"+"Predicted emotion: " + best_guess, fontweight='bold')
	imshow(images,cmap='gray')
	
	right = fig.add_subplot(122)	
	pos = arange(7)+.5    # the bar centers on the y axis
	barh(pos, prediction.tolist()[0], align='center')
	xlim([0, 1])
	yticks(pos, emotion_name[0:7])
	xlabel('Confidence')
	grid(True)
	plt.tight_layout()
	plt.show()

def plot_image_no_pred(images, emotion_num):
	images = np.reshape(images, [48, 48])
	plt.figure().suptitle("correct emotion: " + emotion_name[emotion_num], fontsize=14, fontweight='bold')
	plt.imshow(images, cmap='gray')
	plt.show()

filename_queue = tf.train.string_input_producer(['train.csv'])
reader = tf.TextLineReader(skip_header_lines=1) #skip_header_lines=1
_, csv_row = reader.read(filename_queue)
record_defaults = [[0], [""]]
emotion, pixel_array = tf.decode_csv(csv_row, record_defaults=record_defaults)

emotion_batch, pixel_array_batch = tf.train.shuffle_batch(
      [emotion, pixel_array], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)

prediction = conv_neural_network_model(x)
normalized_prediction = tf.nn.softmax(conv_neural_network_model(x))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train_step = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	## DO A TRAIN
	if (choice == "continue"):
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("NN model has been restored for continued training!")
	if (choice == "train" or choice == "continue"):
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for batch in range(int(n_examples/batch_size)):
				cur_emotion_batch, cur_pixel_array_batch = sess.run([emotion_batch, pixel_array_batch])		
				append_matrix_emotion = list()
				append_matrix_name = list()
				for item in range(batch_size):
					cur_pixel_array_batch[item] = np.fromstring(cur_pixel_array_batch[item], dtype=int, sep=" ")
					append_matrix_emotion.append(cur_pixel_array_batch[item])
					append_matrix_name.append(val_to_one_hot(cur_emotion_batch[item]))
				_, c = sess.run([train_step, cost], feed_dict = {x: np.array(append_matrix_emotion), y: np.array(append_matrix_name)}) #np.reshape(cur_pixel_array_batch[item], [1, 2304])
				epoch_loss += c	
			print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
			saver.save(sess, FLAGS.checkpoint_dir+"model.ckpt", global_step=hm_epochs)
			print("Progress checkpoint saved")
			
			
		## DO AN ACCURACY PRINT
		cur_emotion_batch, cur_pixel_array_batch = sess.run([emotion_batch, pixel_array_batch])	
		accuracy = 0	
		append_matrix_emotion = list()
		append_matrix_name = list()	
		for item in range(batch_size):
			cur_pixel_array_batch[item] = np.fromstring(cur_pixel_array_batch[item], dtype=int, sep=" ")
			value = sess.run(prediction, feed_dict={x: np.array([cur_pixel_array_batch[item]] , dtype=np.float32)})
			if cur_emotion_batch[item] == np.argmax(value[0]):
				accuracy+=1
		print("Correct:", str(accuracy)+"/"+str(batch_size), "Accuracy:", accuracy/batch_size)
		saver.save(sess, FLAGS.checkpoint_dir+"model.ckpt", global_step=hm_epochs)
		print("NN model has been saved.")
	else:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("NN model has been restored!")
		else:
			print("no checkpoint found???")
			exit()
	## DO A VISUALIZE
	cur_emotion_batch, cur_pixel_array_batch = sess.run([emotion_batch, pixel_array_batch])	
	for item in range(min(10, batch_size)):
		cur_pixel_array_batch[item] = np.fromstring(cur_pixel_array_batch[item], dtype=int, sep=" ")
		value = sess.run(prediction, feed_dict={x: np.array([cur_pixel_array_batch[item]] , dtype=np.float64)})
		normalized_value = (value-np.mean(value))/np.std(value)
		plot_image(cur_pixel_array_batch[item], cur_emotion_batch[item], sess.run(tf.nn.softmax(normalized_value)), np.argmax(value[0]))

	## DO A WEBCAM
	face_cascade = cv2.CascadeClassifier('/home/forsythe/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
	cap = cv2.VideoCapture(0)
	#cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
	z = 8 #zoom factor (lower value is higher zoom)
	down_offset = 25
	#final plot
	plt.ion()
	start_time = time.time()
	while True:
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		for (xx,yy,w,h) in faces:
			cv2.rectangle(img,(xx+w//z,yy+h//z+down_offset),(xx+w-w//z,yy+h-h//z+down_offset),(255,0,0),2)
			roi_gray = gray[yy+h//z+down_offset:yy+h-h//z+down_offset, xx+w//z:xx+w-w//z]
			#print(type(roi_gray))
			#cv2.imshow("crop", cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA))
			if time.time() - start_time > 1:
				start_time = time.time()
				cur_pixel_array = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
				#print(cur_pixel_array.shape)
				cv2.imshow("webcam",cur_pixel_array)

				cur_pixel_array = np.resize(cur_pixel_array, (1, 48*48))
				#print(len(cur_pixel_array[0]))
				#print(type(cur_pixel_array[0]))
				value = sess.run(prediction, feed_dict={x: cur_pixel_array})

				normalized_value = (value-np.mean(value))/np.std(value)
				correct_emotion = emotion_name[np.argmax(value[0])]
				best_guess = emotion_name[np.argmax(value[0])]
				normalized_value = sess.run(tf.nn.softmax(normalized_value))

				txt = ""
				for k in range(n_classes):
					txt +=  str(emotion_name[k]) + ": " + str(round(normalized_value[0][k], 3)) + "\n"

				plt.clf()
				plt.title("Predicted emtion: " + best_guess, fontweight='bold')
				plt.barh(range(7), normalized_value.tolist()[0], align='center')	
				yticks(range(7), emotion_name[0:7])
				xlim([0, 1])
				grid(True)
				xlabel('Confidence')
				plt.draw()
				plt.pause(0.0001)

			
		cv2.imshow('img',img)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
	coord.request_stop()
	coord.join(threads)




	

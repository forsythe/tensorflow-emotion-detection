import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

###################constants
emotion_name = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

n_nodes_hl1 = 200
n_nodes_hl2 = 200
n_nodes_hl3 = 200

n_examples = 28709
n_classes = 7

capacity = 2000
batch_size = 500
min_after_dequeue = 1000
hm_epochs = 200


#################
x = tf.placeholder('float', [None, 2304]) #48*48=2304
y = tf.placeholder('float',[None, n_classes])

def neural_network_model(data):
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([2304, n_nodes_hl1])), 
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 
					  'biases': tf.Variable(tf.random_normal([n_classes]))}
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
	return output


def val_to_one_hot(val):
	ans = np.array([0, 0, 0, 0, 0, 0, 0])
	ans[val]=1
	return ans

####################
def plot_image(images, emotion_num, prediction, prediction_best_guess):
	images = np.reshape(images, [48, 48])
	plt.figure().suptitle("correct emotion: " + emotion_name[emotion_num] + "\n" + "best guess: " + emotion_name[prediction_best_guess], fontsize=14, fontweight='bold')
	#print(tf.to_float(prediction[0:1]))
	for k in range(n_classes):
		plt.text(-15, 10+3*k, str(emotion_name[k]) + ": " + str(prediction[0][k]), fontsize=12)
	plt.imshow(images, cmap='gray')
	plt.show()

def plot_image_no_pred(images, emotion_num):
	images = np.reshape(images, [48, 48])
	plt.figure().suptitle("correct emotion: " + emotion_name[emotion_num], fontsize=14, fontweight='bold')
	plt.imshow(images, cmap='gray')
	plt.show()

###################

filename_queue = tf.train.string_input_producer(['train.csv'])
reader = tf.TextLineReader(skip_header_lines=1) #skip_header_lines=1
_, csv_row = reader.read(filename_queue)
record_defaults = [[0], [""]]
emotion, pixel_array = tf.decode_csv(csv_row, record_defaults=record_defaults)
#pixel_array = tf.reshape(tf.strided_slice(tf.decode_raw(pixel_array, tf.uint8), [0], [48*48], [1]), [48, 48])

emotion_batch, pixel_array_batch = tf.train.shuffle_batch(
      [emotion, pixel_array], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)

prediction = neural_network_model(x)
normalized_prediction = tf.nn.softmax(neural_network_model(x))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train_step = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

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
				#print("adding")
				#print(append_matrix_emotion[item])
				#print("adding")
				#print(cur_emotion_batch[item])
				#print(np.array(append_matrix_emotion))
				#print(np.array(append_matrix_name))
				#plot_image(append_matrix_emotion[item], cur_emotion_batch[item])	
			#print("about to train with x:", np.array(append_matrix_emotion).shape)
			#print("about to train with y:", np.array(append_matrix_name).shape)
			_, c = sess.run([train_step, cost], feed_dict = {x: np.array(append_matrix_emotion), y: np.array(append_matrix_name)}) #np.reshape(cur_pixel_array_batch[item], [1, 2304])
			epoch_loss += c	
		print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
		
	## DO AN ACCURACY PRINT
	cur_emotion_batch, cur_pixel_array_batch = sess.run([emotion_batch, pixel_array_batch])	
	accuracy = 0	
	append_matrix_emotion = list()
	append_matrix_name = list()	
	for item in range(batch_size):
		cur_pixel_array_batch[item] = np.fromstring(cur_pixel_array_batch[item], dtype=int, sep=" ")
		#append_matrix_emotion.append(cur_pixel_array_batch[item])
		#append_matrix_name.append(val_to_one_hot(cur_emotion_batch[item]))
		value = sess.run(prediction, feed_dict={x: np.array([cur_pixel_array_batch[item]] , dtype=np.float32)})
		#print("cur emotion is", cur_emotion_batch[item])
		#print("pls be different",value[0])		
		#print("np argmax value is", np.argmax(value[0]))
		if cur_emotion_batch[item] == np.argmax(value[0]):
			accuracy+=1
	print("Correct:", str(accuracy)+"/"+str(batch_size), "Accuracy:", accuracy/batch_size)
	
	###
	## DO A VISUALIZE
	cur_emotion_batch, cur_pixel_array_batch = sess.run([emotion_batch, pixel_array_batch])	
	accuracy = 0	
	append_matrix_emotion = list()
	append_matrix_name = list()	
	for item in range(min(10, batch_size)):
		cur_pixel_array_batch[item] = np.fromstring(cur_pixel_array_batch[item], dtype=int, sep=" ")
		value = sess.run(prediction, feed_dict={x: np.array([cur_pixel_array_batch[item]] , dtype=np.float32)})
		plot_image(cur_pixel_array_batch[item], cur_emotion_batch[item], value, np.argmax(value[0]))

	
	for item in range(0):
		cur_emotion, cur_pixel_array = sess.run([emotion, pixel_array])
		cur_pixel_array = np.fromstring(cur_pixel_array, dtype=int, sep=" ")
		#print("cur pixel array", cur_pixel_array)
		#print("prediction", sess.run(normalized_prediction, feed_dict={x: np.array([cur_pixel_array])}))
		vis_value = sess.run(normalized_prediction, feed_dict={x: np.array([cur_pixel_array])})
		#print("prediction array:", vis_value)
		#print("answer array:", val_to_one_hot(cur_emotion))
		plot_image(cur_pixel_array, cur_emotion, vis_value, np.argmax(vis_value))
	coord.request_stop()
	coord.join(threads)

		
		




#train_neural_network(x)




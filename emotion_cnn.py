import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

###################constants
emotion_name = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral", "unknown"]

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_examples = 28709
n_classes = 7

capacity = 2000
batch_size = 100
min_after_dequeue = 1000
hm_epochs = 10


#################
x = tf.placeholder('float', [None, 2304]) #48*48=2304
y = tf.placeholder('float')

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

result = neural_network_model(x)


def val_to_one_hot(x):
	ans = np.array([0, 0, 0, 0, 0, 0, 0])
	ans[x]=1
	return ans

####################
def plot_image(images, emotion_num=7, prediction=None):
	images = np.reshape(images, [48, 48])
	plt.figure().suptitle("correct emotion: " + emotion_name[emotion_num], fontsize=14, fontweight='bold')
	#print(tf.to_float(prediction[0:1]))
	if not prediction is None:
		for k in range(n_classes):
			plt.text(-15, 10+5*k, str(emotion_name[k]) + ": " + str(prediction[0][k]), fontsize=12)
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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	'''
	for i in range(1): #int(n_examples/batch_size))
		cur_emotion, cur_pixel_array = sess.run([emotion, pixel_array])
		cur_pixel_array = np.resize(np.fromstring(cur_pixel_array, dtype=int, sep=" "), [48, 48])
		
		#plot_image(cur_pixel_array, cur_emotion)
		#print(sess.run(tf.cast(cur_pixel_array, dtype=tf.int16)))
		#cur_pixel_array = tf.reshape(tf.strided_slice(tf.decode_raw(pixel_array, tf.int8), [0], [48*48], [1]), [48, 48])
		print(cur_pixel_array)
		plot_image(cur_pixel_array, cur_emotion)
	'''
	for epoch in range(hm_epochs):
		epoch_loss = 0
		for batch in range(int(n_examples/batch_size)):
			#print("currently on batch", batch)
			cur_emotion_batch, cur_pixel_array_batch = sess.run([emotion_batch, pixel_array_batch])		
			append_matrix_emotion = list()
			append_matrix_name = list()
			for item in range(batch_size):
				#print("\tcurrently on item", item)				
				#cur_pixel_array_batch[item] = np.resize(np.fromstring(cur_pixel_array_batch[item], dtype=int, sep=" "), [48, 48])
				cur_pixel_array_batch[item] = np.fromstring(cur_pixel_array_batch[item], dtype=int, sep=" ")
				append_matrix_emotion.append(cur_pixel_array_batch[item])
				append_matrix_name.append(val_to_one_hot(cur_emotion_batch[item]))
				#print(cur_pixel_array_batch[item])
				#plot_image(cur_pixel_array_batch[item], cur_emotion_batch[item])
			#print(np.array(append_matrix))
			#cur_pixel_array_batch = np.array(cur_pixel_array_batch)
			_, c = sess.run([optimizer, cost], feed_dict = {x: np.array(append_matrix_emotion), y: np.array(append_matrix_name)}) #np.reshape(cur_pixel_array_batch[item], [1, 2304])
				#print(cur_pixel_array_batch[item].shape)
			epoch_loss += c
		print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
	
	## DO A TEST
	cur_emotion_batch, cur_pixel_array_batch = sess.run([emotion_batch, pixel_array_batch])		
	
	for item in range(min(5, batch_size)):
		append_matrix_emotion = list()


		cur_pixel_array_batch[item] = np.fromstring(cur_pixel_array_batch[item], dtype=int, sep=" ")
		append_matrix_emotion.append(cur_pixel_array_batch[item])		

		value = sess.run(result, feed_dict={x: np.array(append_matrix_emotion, dtype=np.float32)})
	
		plot_image(append_matrix_emotion[0], cur_emotion_batch[0], value)

	coord.request_stop()
	coord.join(threads)

		
		




#train_neural_network(x)



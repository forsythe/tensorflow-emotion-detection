import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

filename_queue = tf.train.string_input_producer(['sample.csv']) #  list of files to read

reader = tf.TextLineReader(skip_header_lines=1) #skip_header_lines=1
_, csv_row = reader.read(filename_queue)

record_defaults = [[-1], [""]]
emotion, pixel_array = tf.decode_csv(csv_row, record_defaults=record_defaults)
#features = tf.pack([pixel_array])

def plot_images(images):
	images = np.reshape(images, (48, 48))
	plt.imshow(images, cmap='gray')
	plt.show()

with tf.Session() as sess:
	# Start populating the filename queue.
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for i in range(2):
		# Retrieve a single instance:
		cur_emotion, cur_pixel_array = sess.run([emotion, pixel_array])
		print(cur_emotion, cur_pixel_array)
		cur_pixel_array = np.fromstring(cur_pixel_array, dtype=int, sep=" ")
		plot_images(cur_pixel_array)
	coord.request_stop()
	coord.join(threads)

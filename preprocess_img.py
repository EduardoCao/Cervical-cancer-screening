import numpy

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def my_load_img():
	l_x_train = list()
	l_x_test = list()
	l_y_train = list()
	l_y_test = list()
	f = open("/home/diliu/kaggle/no_crop/train.txt", 'r')
	lines = f.readlines()
	for line in lines:
		if not "jpg" in line:
			continue
		items = line.split('\t')
		if "AType" in items[0].strip(): 
			img = load_img('/home/diliu/kaggle/no_crop/additional/%s/%s.jpg' %(items[0].strip(), items[1].strip()))
		else:
			img = load_img('/home/diliu/kaggle/no_crop/train/%s/%s.jpg' %(items[0].strip(), items[1].strip()))
		x = img_to_array(img)
		# if int(items[0]) < 1200:
		l_x_train.append(x)
		type_of_img = int(items[1].split('_')[1]) - 1
		tmp = list()
		tmp.append(type_of_img)
		l_y_train.append(tmp)
		# else:
			# l_x_test.append(x)
			# type_of_img = int(items[1].split('_')[1]) - 1
			# tmp = list()
			# tmp.append(type_of_img)
			# l_y_test.append(tmp)

	res_train_x = numpy.array(l_x_train)
	res_train_y = numpy.array(l_y_train)
	# res_test_x = numpy.array(l_x_test)
	# res_test_y = numpy.array(l_y_test)

	return res_train_x, res_train_y #, res_test_x, res_test_y

def my_load_test():
	l = list()
	f = open("rectangles_test.csv", 'r')
	lines = f.readlines()
	for line in lines:
		if "image_id" in line:
			continue
		items = line.split(',')
		img = load_img('./output_resize_test_64_64/%s.jpg' %(items[0]))
		x = img_to_array(img)
		l.append(x)
	res = numpy.array(l)
	return res


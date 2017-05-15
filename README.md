<H1>代码说明</H1>
* 源图片存放在./input/train和./input/test等文件夹下
* segmentation.py是图片切分的脚本文件。
	* parallelize\_image\_cropping是针对train set
	* parallelize\_image\_cropping_test是针对test set
	* 生成的是.csv文件，内含需要切割出来的子图片的左上顶点、长宽信息
* crop_pic.py是根据上一步生成的.csv的数据，进行图片切割、缩放的
* preprocess_img.py是用来将图片转化成numpy.array格式，然后做训练集、验证集切割的
* cifar10_cnn.py是利用keras里面的例子，cifar10的cnn的例子modify来得
* run.py和resnet.py是resnet18的model
# SmartPicture
**简单说明**

* Test2_VGG_train_2.py

  使用cifar-10训练数据集训练模型

* Test2_VGG_test_2.py

  使用cifar-10测试数据集测试模型

* net_vgg_train_2.pkl

  模型训练参数保存结果

* Tmp2.py

  使用本地图片测试模型预测结果
    
**9月7日**
* 新增v3版本
* 采用更深层的卷积神经网络
* 识别率更高

**9月8日**
* 新增TinyImageNet
* 可以识别tiny数据集的前10个分类
* 对于tiny数据集的分类正确率可达99%

**9月8日**
* 从tiny数据集中整理出常见的23个标签数据
* 使用标签数据训练出训练集准确率为95%左右的网络
* tiny_train_2  训练      
* tiny_test_2   读取几张网上随便找的图片进行识别
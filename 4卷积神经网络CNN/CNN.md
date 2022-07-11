# 4. 卷积神经网络CNN

### 参考：

​	https://www.zhihu.com/question/22298352

​	https://blog.csdn.net/suiyueruge1314/article/details/104949254

​	Ian Goodfellow, Yoshua Bengio and Aaron Courville: Deep Learning

## 4.1 基础知识介绍

### 4.1.1 卷积

​	我们知道深度学习是一种表示学习，通过神经网络来提取数据中的特征，卷积神经网络也是如此，而卷积操作就是CNN提取数据体征的手段。

####  1. 卷积的定义 

​	卷积在数学中是一种特殊的运算，表示为$s(t)=(x*w)(t)$，x为输入，函数w为核函数，输出被称为特征映射(feature map)，它的离散表示被定义为：![image-20210503175508437](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210503175508437.png)

其含义是：核函数在原函数上滑动，特征映射输出中的每个值即每次滑动中核函数与原函数对应值相乘之和。注意到卷积公式，我们对核进行了翻转在与输入相乘，但在神经网络中我们省去了翻转，事实上未翻转的运算被称为相关，如下图：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708113235088.png" alt="image-20210708113235088" style="zoom:80%;" />	

卷积同样可用于二维数据，如图像$I$，与二维的核$K$进行卷积：![image-20210503175516460](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210503175516460.png)



![img](https://pic3.zhimg.com/50/v2-15fea61b768f7561648dbea164fcb75f_hd.webp?source=1940ef5c)	

![image-20210708112856145](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708112856145.png)

#### 2. 卷积的含义

​	那么为什么卷积网络中使用卷积可以提取图像信息呢？我们从图像处理角度理解这个问题。我们知道图像就是一个矩阵，当我们使用不同的核（滤波器）进行卷积时，我们可以得到包含不同信息的输出。例如我们有图片：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210503174906511.png" alt="image-20210503174906511" style="zoom:80%;" />

​	以及卷积核：[[-1,0,1],[-1,0,1],[-1,0,1]]，它的意义是图像左边的像素减去右边的像素，也就是图像水平方向的梯度。当应用此卷积核，我们便得到了图像竖直方向的纹理信息：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210503175051782.png" alt="image-20210503175051782" style="zoom:80%;" />



​	同样，我们使用卷积核[[-1,-1,-1],[0,0,0],[1,1,1]]，就可以得到水平纹理信息

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210503175132284.png" alt="image-20210503175132284" style="zoom:80%;" />

​	当我们有许多不同的卷积核时，我们就可以从图像中提取出图像隐藏的不同信息，在卷积网络中存每个神经元都是不同的卷积核，提取不同的图像信息，这也是为什么卷积网络可以拟合图像数据，每个卷积核卷积得到的输出叫做**特征图 feature map**。

![image-20210708113151961](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708113151961.png)



#### 3. 卷积的优点：

	1. 保持了空间结构：卷积的滑动窗口的计算方式可以保留原数据的空间信息，如图片经过二维卷积仍是二维矩阵；

![image-20210708174854518](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708174854518.png)

	2. 减少参数量：对于$28\times 28\times 3$大小的图片，如果使用全连接层进行10分类，那么需要的参数量是$28\times 28\times 3\times 10$，而如果使用10个$3\times 3$的卷积核，仅需要$10\times 3\times 3\times 3$的参数。

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708175308866.png" alt="image-20210708175308866" style="zoom:80%;" />

#### 4. 卷积的特点

	1. 离散连接 Sparse connections：单个卷积的输出仅与特定位置的输入有关，相比全连接网络，每个输出都与全部输入有关；

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708180007939.png" alt="image-20210708180007939" style="zoom:80%;" />

	2. 参数共享 Parameter sharing：单个卷积核被多个输入共享，例如下图，对于加粗权重由于滑动，被所有输入共享，而全连接网络每个输入输出的权重都是一一对应的；

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708180309478.png" alt="image-20210708180309478" style="zoom:80%;" />

	3. 等变表示 Equivariant representations：对输入的空间变换有泛化性。

#### 5. 相关概念

##### 	5.1 感受野 receptive filed：

​		一个卷积核能跨越（看到）的输入的大小，在网络中感受野逐渐增大：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708193919678.png" alt="image-20210708193919678" style="zoom:80%;" />

##### 	5.2 步长 stride

​		在卷积的过程中，我们有时会为了减少计算量，改变卷积过程中滑动窗口每次跨越的距离，如图卷积的步长为2：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708194101002.png" alt="image-20210708194101002" style="zoom:80%;" />

##### 	5.3 填充 padding

​		如果我们不对卷积的输入做任何操作，那么对于输入大小为6，卷积核大小为3的卷积输出大小为4，通常为了消除这样的输出尺度减小或人为地控制输出的大小，我们会对卷积输入进行填充，最常用的是**0填充(zero padding)**。如下图，使用0对上层输入进行填充，使所有的输出都是相同的大小：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708194537926.png" alt="image-20210708194537926" style="zoom:80%;" />

​	有三种填充方式：



​	（1） Valid Padding：不做填充，输入m，卷积核k，输出大小为m-k+1;

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708195039387.png" alt="image-20210708195039387" style="zoom:30%;" />

​	（2）Same Padding：进行填充，使输入m，输出大小也为m；

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708195025479.png" alt="image-20210708195025479" style="zoom:30%;" />

​	（3）Full Padding：每个输入都对等量的输出有贡献，输入m，输出m+k-1

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708195010015.png" alt="image-20210708195010015" style="zoom:30%;" />

### 4.1.2 池化

#### 1. 池化的定义

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210503175628499.png" alt="image-20210503175628499" style="zoom:80%;" />

在卷积神经网络中，除了卷积层，还有一个不可或缺的部分叫做池化层(pooling layer)，其主要目的是尽量保留原有信息的同时，将数据的尺寸降低。池化是一种运算，它不包含任何参数，池化有许多形式，如平均池化，最大池化等，我们介绍**最大池化**。

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210503180136665.png" alt="image-20210503180136665" style="zoom:80%;" />

如图所示，在原图像上使用（2,2）的最大池化，即在每个（2,2）范围中选出最大值作为输出，图像的尺寸也被缩小为1/2。

#### 2. 池化的意义

​	池化的意义除了可以缩减输出尺寸外，池化还具有局部等变性，因为池化操作只选择局部中的特定值，所以输入的局部变化不会影响池化结果，比如人的眼睛总是在人脸的上方，但有点人眼睛偏高一些，有的人偏低，通过池化我们总是能挑出人脸上方的眼睛，这不会收到其局部位置的差别而改变。

1. 不变性：

   （1） 使网络对输入的小变换的鲁棒性更高，意思是尽管输入进行了一些改变，网络的输出仍是不变的；

   （2） 局部特征具有一定的空间不变性，如上文所述；

   （3）池化可以被看做一个无限强的先验，令网络必须对微小变换具有不变性。

2. 高效性：池化可以将k个单元压缩为1个单元，减少了网络的计算量和内存要求

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708191644062.png" alt="image-20210708191644062" style="zoom:80%;" />

## 4.2 卷积神经网络 CNN

### 4.2.1 卷积神经网络概述

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708192744803.png" alt="image-20210708192744803" style="zoom:80%;" />	

​	卷积神经网络与多层感知机相似，网络由最基本的模块（卷积层、非线性激活层、池化层）重复堆叠而成，在网络的末尾，针对我们的任务的不同，会拼接上不同的输出层，比如如果是分类任务，其输出层就是全连接层+softmax层。

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708193021240.png" alt="image-20210708193021240" style="zoom:80%;" />

​	

在网络逐渐深入的过程中，网络从数据中提取的信息逐渐从低层次语义信息，变为高级语义信息：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708193159798.png" alt="image-20210708193159798" style="zoom:80%;" />

### 4.2.2 卷积网络的多层卷积和3D特征映射

​	在卷积网络中，一个卷积层的输入通常是3维的$(C\times H\times W)$，例如一张彩色图片包含3个通道，其大小为$3*32*32$；

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708195907703.png" alt="image-20210708195907703" />

​	而一个卷积层包含多个卷积核，每个卷积核对输入进行卷积可以得到一个通道的输出，因此卷积层的输出也是3维的$(C\times H\times W)$，其中通道数$C$对应卷积层的卷积核数量：

![image-20210708200030708](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708200030708.png)

​	为了令卷积核可以学习到多个通道的信息，通常我们使用3维的卷积核，大小为$(in\_channels, h, w)$，in_channels对应上层输出的通道数，所以单个卷积层的参数量为卷积核数量*卷积核大小：$(out\_channels, in\_channels, h, w)$

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708200525683.png" alt="image-20210708200525683" style="zoom:80%;" />

### 4.2.3 经典的卷积神经网络

#### 1. AlexNet

![image-20210708200701934](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708200701934.png)

#### 2. VGGNet

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708200748137.png" alt="image-20210708200748137" style="zoom:80%;" />

#### 3. GoogLeNet

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708200825170.png" alt="image-20210708200825170" style="zoom:80%;" />

#### 4. ResNet

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708200858035.png" alt="image-20210708200858035" style="zoom:80%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708200920154.png" alt="image-20210708200920154" style="zoom:80%;" />

#### 5. DenseNet

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210708200949735.png" alt="image-20210708200949735" style="zoom:80%;" />
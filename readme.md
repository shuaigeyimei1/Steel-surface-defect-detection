# 一.整体概述

## 1.数据集介绍

我们选用的是NEU-DET钢材表面缺陷数据集，其中训练样本共1770张图片（200×200），验证样本共24张图片。

缺陷类型分为'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'共六种

![image-20230621192003302](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192003302.png)

针对次数据集，我们要完成的任务有两个。

1.用方框框出图片中的所有缺陷，对缺陷进行正确定位。2.正确识别出框出的缺陷类别。

由于下载下来的数据集的标签文件是xml格式，而我们使用的Yolov5算法要求输入的数据标签格式为txt格式。所以我们通过上网搜索，用了一个脚本文件对格式进行了转换。

转换后的标签文件内容如图所示

![image-20230621192019610](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192019610.png)

图中 第一列代表缺陷的类型，

第二、三列代表缺陷框中心点的x、y坐标，第四、五列代表缺陷框的宽度和高度(归一化之后的结果)

行数代表这张图中有几个缺陷

## 2.算法介绍

我们采用Yolov5算法进行数据集的训练和验证。

”You only look once“，相比于其他目标检测算法，Yolo采用单阶段检测的方法，可以在一次前向计算中完成目标检测，具有更快的检测速度和较高的检测精度。

Yolo为了适应更多的场合，提供了多个版本，也就是n、s、m、l、x。类比于衣服的大小号，从n到x，模型架构层数越来越多，速度越来越慢但训练效果也会越来越好。

由于电脑硬件条件的限制，这里我们采用s号的模型进行数据集的训练。

# 二.数据预处理

为了使网络训练效果的更好，除了对数据集进行一些已有的常规数据预处理以外（如图片缩放、剪切、翻转、旋转等），我们在对数据处理层面上进行了一系列的改进，使得在训练时可以对数据有更充分地利用。而且只是增加了一定的训练成本，不影响测试。

## 1.Mosaic data augmentation

参考CutMix的做法，我们使用马赛克数据增强策略，将四张图像拼接成一张进行训练，可以增加机器学习训练集的难度，并间接增加了batchsize的大小，使得单GPU就能训练的很好，提高了模型的鲁棒性和泛化能力

如图，这里的batchSize为4，所以一共有四张大图，而每张大图中又是由4张小图拼接而成，所以实际的batchSize可以视为16

![image-20230621192034298](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192034298.png)

## 2.DropBlock

传统的神经网络经常会使用Dropout的正则化方法在训练过程中随机杀死一些神经元，来达到防止过拟合的目的。但是由于这些神经元是散落在各处的，在神经网络对图像进行类别判断时，可以根据周围的像素点进行推断，所以这并不能有效的提高神经网络判断出类别的难度。

我们抛弃传统的随机失活方式，采用DropBlock的方法，随机杀死一个大区域中的所有神经元，这样训练出来的网络具有更强的泛化能力。

![image-20230621192049312](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192049312.png)

## 3.Label smoothing标签平滑

在处理一个分类问题时，传统的训练集通常以（0，1）的形式来打标签，是该类别则为1，否则为0。这样的表示方法过于绝对，使得网络更容易过拟合。

我们采用标签平滑的方式，可能就会将（0，1）这样的标签通过一定的公式改为（0.05，0.95），增大了网络的抗过拟合能力。使用后每个类别的簇内距离更紧密，而簇间距离更大了（下图右）

![image-20230621192059360](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192059360.png)

# 三.网络模型

## Backbone骨干网络

![image-20230621192111509](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192111509.png)

1.Conv模块

Conv卷积层用于提取输入特征中的局部空间信息。该模块采用了CSPNet（Cross-Stage Partial Network）网络结构。  
CSP网络结构是由两个相同的部分组成，每个部分包含一个卷积层和一个残差块，两个部分之间还有一个跨阶段的连接。可以将输入特征图分成两部分，分别进行卷积和残差块的处理，然后再将两部分的特征图合并起来。这样可以减少计算量和参数量，提高网络的效率和速度。

![image-20230621192120272](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192120272.png)

这里使用的激活函数为Mish激活函数。相比于传统的ReLU激活函数，在输入值为负数时，将一定范围内的特征值予以保留。具有更好的非线性特征提取能力，有助于提高模型的表现力和泛化能力。

![image-20230621192127072](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192127072.png)

2.C3模块

C3模块是网络中的一个重要组成部分，其主要作用是增加网络的深度和感受野，提高特征提取的能力。

C3模块是由三个Conv块构成的，其中第一个Conv块的步幅为2，可以将特征图的尺寸减半，第二个Conv块和第三个Conv块的步幅为1。C3模块中的Conv块采用的都是3x3的卷积核。在每个Conv块之间，还加入了BN层和LeakyReLU激活函数，以提高模型的稳定性和泛化性能。

3.SPPF模块

SPPF一种金字塔池化结构，可以对不同大小的特征图进行池化，从而增强模型对不同尺度目标的感知能力。其先通过一个标准卷积模块将输入通道减半，然后分别做kernel-size为5，9，13的maxpooling（对于不同的核大小，padding是自适应的）。

对三次最大池化的结果与未进行池化操作的数据进行concat，最终合并后channel数是原来的2倍。

![image-20230621192142019](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192142019.png)

## 输出端:Head

![image-20230621192203063](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192203063.png)

随着卷积层的增多，特征图越来越小，特征图中一个特征点的感受野越来越大，相当于能看到的原始图像的范围越大，更适合检测大目标。所以感受野小、中、大的特征图，分别适合用来检测小目标、中目标和大目标。三种特征图的候选框的尺寸也依次增大，每种特征图共三个候选框。

但是仅用感受野小的和中的两个特征图来对小、中目标进行检测效果可能并不好，因为他的训练的语义信息不如最后的特征图丰富。解决办法是：一方面感受野大的特征图通过上采样（Upsample）的方法与感受野小的特征图进行特征融合，像老者给年轻人传授经验；另一方面，引入了一条自底向上的路径（PAN），使底层信息更容易的传到顶层

![image-20230621192215906](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192215906.png)

# 四.结果展示与性能评估

我们使用训练集对Yolov5s模型共进行了120轮训练，batchSize为6，以下是我们对结果的展示及各性能指标的分析

## 训练结果

**训练集所有标签的统计**

各类别的数量（左上），缺陷在图中的位置分布（左下），缺陷的尺寸（右下）

![image-20230621192228126](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192228126.png)

**precision精度：**所有预测为正样本中有多少确实是正样本

![image-20230621192349356](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192349356.png)

**recall召回率：**所有实际为正样本的有多少被检测出来了

![image-20230621192356407](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192356407.png)

**mAP平均精度：**

将精度和召回率两个指标综合起来。recall会随着预测出的样本数量而增加，直到把所有正样本全部预测出来，recall为1，但precision会上下波动。从而我们可以在坐标轴上画出precision和recall的曲线，AP为precision的平均值，mAP是所有类别的AP的平均值。

下图为置信度阈值为0.5时的mAp图像

![image-20230621192403553](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192403553.png)

下图为置信度阈值分别为0.5-0.95时的平均值

![image-20230621192410833](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192410833.png)

**损失函数：**

**位置误差**

![image-20230621192318857](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192318857.png)

**置信度误差**

![image-20230621192327557](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192327557.png)

**分类误差**

![image-20230621192423831](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192423831.png)

**总结**

从上述图片中可以看出，我们训练出的模型效果比较好。在第103轮模型收敛到最佳情况，精度为0.75，召回率0.82，mAP_0.5为0.85

![image-20230621192431107](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192431107.png)

## 模型验证

我们使用第103轮的训练权重数据来对模型进行验证，输入验证集的图片，置信度阈值设为0.8（表示置信度大于0.8的预测框才会显示在图中），得到下图的输出结果

![image-20230621192438032](C:\Users\G5\AppData\Roaming\Typora\typora-user-images\image-20230621192438032.png)

与验证集的标签进行比对，发现仅有一处误检，没有漏检，模型效果较为满意

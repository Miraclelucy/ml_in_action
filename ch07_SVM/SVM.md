
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/1.png?raw=true)  
Support Vector Machine支持向量机（SVM）是一个非常强大机器学习模型，能够执行线性或非线性的分类，回归，异常值检测等，用于人脸检测，图像分类，文本分类等非常广泛的领域。它是经典机器学习算法中最流行的模型之一。

### 一、线性SVM

假设我们有两类数据，我们要使用SVM进行分类，如下图所示：  
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/2.png?raw=true)

这两个类别可以用直线（线性可分）分离。而且我们可以画出无数条分隔线。但是每条线的分隔（预测）能力是不一样的，哪条线的分隔能力最好呢？在左图中，我们可以凭直观判断蓝线比红线分隔数据好，因为2个类别的点离它都比较远。我们常常将这条分割线称为超平面hyperplane，因为在现实中我们工作的目标常是多维数据，多维空间的情况下自然就成为超平面了。所以我们希望找到“最优超平面”。但什么是最优呢? 这背后的原理是什么？通俗来讲，好的划分就是明显区分两类数据，也就是尽量使两类数据对称，还能尽量使未来的数据也能正确的分类，也就是泛化能力要好。

SVM算法就是尝试寻找一个最优的决策边界（最优划分超平面），使得该边界与两个类别最近的样本点距离最远(那些最近的样本点称为“支持向量”)。在这种情况下，我们认为该分类器的分类效果达到最优。SVM即支持向量机，其含义是通过支持向量来计算“最优超平面”后所获得的分类器。其中“机”的意思是机器，可以理解为分类器。  

![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/3.png?raw=true)
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/4.png?raw=true)

### 二、软间隔

如果我们严格要求样本点不得出现在中间的“分隔区域”里，这就是所谓的硬间隔分类。而硬间隔分类有2个问题：只有数据线性分离才有效，对异常值太敏感。  
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/5.png?raw=true)

在上面的数据分类图中，有一个蓝色的异常值。如果对该数据集使用硬间隔分类，我们得到的决策边界的距离将会非常小。为了避免这些问题，最好使用更具弹性的模型。目的是在“保持分隔区间尽可能大”和“不允许样本点过界”之间找到一个良好的平衡，就是允许出现样本点最终可能在分割区间中间甚至出现在另一侧的情况，这称为软间隔分类。如果我们对该数据集应用软间隔分类，则我们将得到比硬间隔分类更大决策边界，模型的泛化能力将会更强。为了度量这个间隔软到何种程度，我们为每个样本引入一个松弛变量 ξ ，令 ξ>=0 ，
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/6.png?raw=true)

### 三、非线性SVM

上面讨论的硬间隔和软间隔都是在说样本完全线性可分或者大部分样本点线性可分。但我们常常会碰到的情况是样本点不是线性可分的，比如这样的：  
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/7.jpg?raw=true)

这种情况的解决方法，就是将二维线性不可分样本映射到高维空间中，让样本点在高维空间线性可分，如下图所示  
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/8.jpg?raw=true)
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/9.png?raw=true)
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/10.png?raw=true)
![avatar](https://github.com/Miraclelucy/ml_in_action/blob/main/img/ch07/11.png?raw=true)

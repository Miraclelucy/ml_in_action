### White Wine Quality白葡萄酒品质数据集的简介
White Wine Quality白葡萄酒品质数据集，经常被用于回归或分类建模的简单而干净的练习数据集。  
有两个样本:
winequality-red.csv:红葡萄酒样本
winequality-white.csv:白葡萄酒样本
这两个数据集与葡萄牙“Vinho Verde”葡萄酒的红色和白色变种有关。
由于隐私和逻辑问题，只有物理化学（输入）和感官（输出）变量可用（例如，没有关于葡萄类型、葡萄酒品牌、葡萄酒售价等的数据）。  
这些数据集可以被视为分类或回归任务。 等级是有序的并且不平衡（例如，普通葡萄酒比优质或劣质葡萄酒多得多）。 
异常值检测算法可用于检测少数优秀或劣质的葡萄酒。 此外，我们不确定是否所有输入变量都是相关的。 因此，测试特征选择方法可能会很有趣。
每个样本都有得分从1到10的质量评分，以及若干理化检验的结果

###	理化性质	字段名称
1	固定酸度	fixed acidity
2	挥发性酸度	volatile acidity
3	柠檬酸	citric acid
4	残糖	residual sugar
5	氯化物	chlorides
6	游离二氧化硫	free sulfur dioxide
7	总二氧化硫	total sulfur dioxide
8	密度	density
9	PH值	pH
10	硫酸盐	sulphates
11	酒精度	alcohol
12	质量	quality

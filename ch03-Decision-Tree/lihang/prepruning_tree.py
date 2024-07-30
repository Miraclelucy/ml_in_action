from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import pandas as pd
#--------数据加载-----------------------------------
iris = load_iris()                          # 加载数据
X = iris.data
y = iris.target
#-------用最优参数训练模型------------------
clf = tree.DecisionTreeClassifier(random_state=0,max_depth=4,min_samples_leaf=10)
clf = clf.fit(X, y)
depth = clf.get_depth()
leaf_node = clf.apply(X)
#-----观察各个叶子节点上的样本个数---------
df  = pd.DataFrame({"leaf_node":leaf_node,"num":np.ones(len(leaf_node)).astype(int)})
df  = df.groupby(["leaf_node"]).sum().reset_index(drop=False)
df  = df.sort_values(by='num').reset_index(drop=True)
print("\n==== 树深度：",depth," ============")
print("==各个叶子节点上的样本个数：==")
print(df)

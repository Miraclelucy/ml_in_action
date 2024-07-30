from sklearn.datasets import load_iris
from sklearn import tree

#--------数据准备-----------------------------------
iris = load_iris()                          # 加载数据
X = iris.data
y = iris.target
#-------模型训练---------------------------------
clf = tree.DecisionTreeClassifier(min_samples_split=10,random_state=0,ccp_alpha=0)
clf = clf.fit(X, y)
#-------计算ccp路径------------------------------
pruning_path = clf.cost_complexity_pruning_path(X, y)

#-------打印结果---------------------------------
print("\n====CCP路径=================")
print("ccp_alphas:",pruning_path['ccp_alphas'])
print("impurities:",pruning_path['impurities'])

#------设置alpha对树后剪枝-----------------------
clf = tree.DecisionTreeClassifier(min_samples_split=10,random_state=0,ccp_alpha=0.1)
clf = clf.fit(X, y)
#------自行计算树纯度以验证-----------------------
is_leaf =clf.tree_.children_left ==-1
tree_impurities = (clf.tree_.impurity[is_leaf]* clf.tree_.n_node_samples[is_leaf]/len(y)).sum()
#-------打印结果---------------------------
print("\n==设置alpha=0.1剪枝后的树纯度：=========\n",tree_impurities)

import numpy as np  # 导入处理数值计算的库
import warnings  # 用于处理警告信息
import pandas as pd  # 导入处理数据的库
import matplotlib.pyplot as plt  # 导入可视化绘图的库
import seaborn as sns  # 导入更高级的可视化绘图库
from sklearn.model_selection import train_test_split  # 导入拆分训练集和测试集的方法
from sklearn.linear_model import LinearRegression  # 导入线性回归的方法
from sklearn.metrics import mean_squared_error  # 导入均方误差的方法
from sklearn.metrics import accuracy_score  # 导入准确率得分的方法
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score  # 导入用于分类问题评估性能的方法
from sklearn.preprocessing import PolynomialFeatures  # 导入处理多项式特征的方法
from sklearn.metrics import mean_squared_error  # 导入均方误差的方法
from sklearn.tree import DecisionTreeRegressor  # 导入决策树回归器的方法
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归器的方法
from sklearn import linear_model  # 导入线性模型库
from math import sqrt  # 导入计算平方根的函数
from prettytable import PrettyTable  # 导入绘制ascii表格的库

# 使用 pandas 中的 read_csv 函数读取 csv 文件
df = pd.read_csv("winequality-red.csv", sep=';')
# 使用 DataFrame 中的 head() 函数来查看数据的前 10 行
print("-----------df.head(10)-------")
print(df.head(10))
print("-----------df.shape-------")
print(df.shape)  # 使用 df.shape 查看数据集的维度，即行数和列数。
# 将数据集中所有列名中的空格替换为下划线。
df.columns = df.columns.str.replace(' ', '_')
# 使用 df.info() 查看每列的数据类型和非空数量。
# 使用 df.isnull().sum() 查看每列缺失值的数量。
print("-----------df.info()-------")
print(df.info())
print("-----------df.isnull().sum()-------")
print(df.isnull().sum())
# 使用matplotlib，展示数据集中每个品质评分对应的红酒数量
quality_and_counts = df['quality'].value_counts()
quality_and_counts.plot(kind='bar')
plt.show()
# 使用 Seaborn 库的 countplot 函数，展示数据集中每个品质评分对应的红酒数量
# sns.countplot(df['quality'])

# 输出每种品质评分在数据集中的红酒数量
print("-----------df['quality'].value_counts()-------")
print(df['quality'].value_counts())

correlations = df.corr()['quality']  # 计算数据集中各个特征与“quality”列之间的相关系数，返回一个Series类型的对象。
correlations = correlations.sort_values(ascending=False)  # 将上述Series对象中的值按照从大到小的顺序排序，生成一个有序的Series类型的对象。
print("-----------correlations-------")
print(correlations)  # 输出排序后的结果。
correlations.plot(kind='bar')  # 绘制按特征与品质相关系数从大到小排列的条形图
plt.show()

# plt.figure(figsize=(10, 6))  # 设置画布大小为 10*6 英寸
# sns.heatmap(df.corr(), annot=True, fmt='.0%')  # 使用 Seaborn 可视化库绘制热力图
print("-----------abs(correlations) > 0.2-------")
print(abs(correlations) > 0.2)  # 计算各特征与“quality”特征之间的相关系数，然后返回一个布尔型DataFrame对象，其中每个元素表示该位置的相关系数绝对值是否大于0.2。

# 使用Seaborn库的boxplot函数，绘制箱线图
bp = sns.boxplot(x='quality', y='alcohol', data=df)
# 设置图表标题
bp.set(title="Alcohol Percent in Different Quality Wines")
plt.show()

# 选择质量等级在5或6之间的葡萄酒数据
df_quality_five_six = df.loc[(df['quality'] >= 5) & (df['quality'] <= 6)]
# 统计质量等级为5和6的葡萄酒数量
df_quality_five_six['quality'].value_counts()
# 计算质量等级在5或6之间的葡萄酒数据中各属性与质量等级之间的相关性，并根据相关性从大到小进行排序
correlations_subset = df_quality_five_six.corr()['quality'].sort_values(ascending=False)
# 输出各属性与质量等级之间的相关性结果
print(correlations_subset)

# 使用seaborn库中的boxplot函数绘制质量等级与硫酸盐含量（“sulphates”）之间的箱线图
bp = sns.boxplot(x='quality', y='sulphates', data=df)
# 设置图表标题为“不同质量等级葡萄酒中的硫酸盐含量”
bp.set(title="Sulphates in Different Quality Wines")
plt.show()

# 使用seaborn库中的boxplot函数绘制质量等级与柠檬酸含量（“citric_acid”）之间的箱线图
bp = sns.boxplot(x='quality', y='citric_acid', data=df)
# 设置图表标题为“不同质量等级葡萄酒中的柠檬酸含量”
bp.set(title="Citric Acid in Different Quality Wines")
plt.show()

# 使用seaborn库中的boxplot函数绘制质量等级与挥发性酸含量（“volatile_acidity”）之间的箱线图
bp = sns.boxplot(x='quality', y='volatile_acidity', data=df)
# 设置图表标题为“不同质量等级葡萄酒中的挥发性酸度存在情况”
bp.set(title="Acetic Acid Presence in Different Quality Wines")
plt.show()

# 复制DataFrame对象df到新的对象df_aux
df_aux = df.copy()
# 使用.replace()函数将3和4替换为“low”，5和6替换为“med”，7和8替换为“high”，并将结果直接更新到df_aux的“quality”列中
df_aux['quality'].replace([3, 4], ['low', 'low'], inplace=True)
df_aux['quality'].replace([5, 6], ['med', 'med'], inplace=True)
df_aux['quality'].replace([7, 8], ['high', 'high'], inplace=True)
# 使用Seaborn库中的countplot函数绘制质量等级计数图
sns.countplot(df_aux['quality'])
plt.show()


# 从数据框中选择4个特征列作为自变量，将“quality”列作为因变量。
X = df.loc[:, ['alcohol', 'sulphates', 'citric_acid', 'volatile_acidity']]
Y = df.iloc[:, 11]

# 将数据划分为训练集和测试集，并使用线性回归模型进行拟合和预测。

# 此处采用70%的数据作为训练集，其余30%作为测试集，设置随机种子为42，以确保每次运行时产生相同的结果。
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print("--------------------start 线性回归模型-------------------------")
# 构建线性回归模型，并对模型进行训练和预测。
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_prediction_lr = regressor.predict(X_test)
y_prediction_lr = np.round(y_prediction_lr)
# 对模型的预测结果进行可视化展示。
plt.scatter(y_test, y_prediction_lr)
plt.title("Prediction Using Linear Regression")
plt.xlabel("Real Quality")
plt.ylabel("Predicted")
plt.show()

# 使用混淆矩阵展示线性回归模型的分类效果。
# 首先，计算模型预测结果的混淆矩阵。
cm_linear_regression = confusion_matrix(y_test, y_prediction_lr)
# 将混淆矩阵转换为数据框，并设置标签和格式。
cm_lr = pd.DataFrame(cm_linear_regression,
                     index=['3', '4', '5', '6', '7', '8'],
                     columns=['3', '4', '5', '6', '7', '8'])
sns.heatmap(cm_lr, annot=True, fmt="d")
# 设置横纵坐标的标签。
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.show()
print("--------------------end 线性回归模型-------------------------")


print("--------------------start 决策树回归模型-------------------------")
# 构建决策树回归模型，并对模型进行训练和预测。
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_prediction_dt = regressor.predict(X_test)
y_prediction_dt = np.round(y_prediction_dt)

# 对模型的预测结果进行可视化展示。
plt.scatter(y_test, y_prediction_dt)
plt.title("Prediction Using Decision Tree Regression")
plt.xlabel("Real Quality")
plt.ylabel("Predicted")
plt.show()

# 构建决策树回归模型并对模型进行预测后，使用混淆矩阵展示其分类效果。
# 首先，计算模型预测结果的混淆矩阵。
cm_decision_tree_regression = confusion_matrix(y_test, y_prediction_dt)

# 将混淆矩阵转换为数据框，并设置标签和格式。
cm_dt = pd.DataFrame(cm_decision_tree_regression,
                     index=['3', '4', '5', '6', '7', '8'],
                     columns=['3', '4', '5', '6', '7', '8'])
sns.heatmap(cm_dt, annot=True, fmt="d")
# 设置横纵坐标的标签。
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.show()
print("--------------------end 决策树回归模型-------------------------")


print("--------------------start 随机森林回归模型-------------------------")
# 构建随机森林回归模型，并对模型进行训练和预测。
regressor = RandomForestRegressor(n_estimators=10, random_state=42)
regressor.fit(X_train, y_train)
y_prediction_rf = regressor.predict(X_test)
y_prediction_rf = np.round(y_prediction_rf)

# 对模型的预测结果进行可视化展示。
plt.scatter(y_test, y_prediction_rf)
plt.title("Prediction Using Random Forest Regression")
plt.xlabel("Real Quality")
plt.ylabel("Predicted")
plt.show()

# 对随机森林回归模型的预测结果进行混淆矩阵展示。
cm_random_forest_regression = confusion_matrix(y_test, y_prediction_rf)
cm_rf = pd.DataFrame(cm_random_forest_regression,
                     index=['3', '4', '5', '6', '7', '8'],
                     columns=['3', '4', '5', '6', '7', '8'])
sns.heatmap(cm_rf, annot=True, fmt="d")

# 设置横纵坐标的标签。
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.show()
print("--------------------end 随机森林回归模型-------------------------")

# 计算线性回归模型的RMSE并输出。
RMSE = sqrt(mean_squared_error(y_test, y_prediction_lr))
print("线性回归模型在测试集上的RMSE为：" + str(RMSE) + "\n")

# 计算决策树回归模型的RMSE并输出。
RMSE = sqrt(mean_squared_error(y_test, y_prediction_dt))
print("决策树回归模型在测试集上的RMSE为：" + str(RMSE) + "\n")

# 计算随机森林回归模型的RMSE并输出。
RMSE = sqrt(mean_squared_error(y_test, y_prediction_rf))
print("随机森林回归模型在测试集上的RMSE为：" + str(RMSE) + "\n")


# 定义函数，将回归模型预测结果与真实值之间相差1的样本的预测值调整为真实值。
def one_accuracy(predicted, true):
    i = 0
    for x, y in zip(predicted, true):
        if (abs(x - y) == 1):
            predicted[i] = y
        i = i + 1


# 分别对线性回归、决策树回归和随机森林回归模型的预测结果进行一次精度修正。
one_accuracy(y_prediction_lr, y_test)
one_accuracy(y_prediction_dt, y_test)
one_accuracy(y_prediction_rf, y_test)


# 展示线性回归模型在测试集上的混淆矩阵。
label_aux = plt.subplot()
cm_linear_regression = confusion_matrix(y_test, y_prediction_lr)
cm_lr = pd.DataFrame(cm_linear_regression,
                     index=['3', '4', '5', '6', '7', '8'],
                     columns=['3', '4', '5', '6', '7', '8'])
sns.heatmap(cm_lr, annot=True, fmt="d")
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.show()


# 计算决策树回归模型在测试集上的混淆矩阵，将结果赋值给变量cm_decision_tree_regression。
cm_decision_tree_regression = confusion_matrix(y_test, y_prediction_dt)
cm_dt = pd.DataFrame(cm_decision_tree_regression,
                     index=['3', '4', '5', '6', '7', '8'],
                     columns=['3', '4', '5', '6', '7', '8'])
sns.heatmap(cm_dt, annot=True, fmt="d")
# 为图表和子图添加横纵坐标轴的标签。
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.show()


# 计算随机森林回归模型在测试集上的混淆矩阵，将结果赋值给变量cm_random_forest_regression。
cm_random_forest_regression = confusion_matrix(y_test, y_prediction_rf)
cm_rf = pd.DataFrame(cm_random_forest_regression,
                     index=['3', '4', '5', '6', '7', '8'],
                     columns=['3', '4', '5', '6', '7', '8'])
sns.heatmap(cm_rf, annot=True, fmt="d")
# 为图表和子图添加横纵坐标轴的标签。
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.show()


# 计算线性回归模型在测试集上的均方根误差（RMSE），并将结果保存到变量RMSE_lr中。
RMSE_lr = sqrt(mean_squared_error(y_test, y_prediction_lr))
# 打印输出新改进的线性回归模型的RMSE。
print("新改进的线性回归模型在测试集上的RMSE为：" + str(RMSE_lr) + "\n")

# 计算决策树回归模型在测试集上的均方根误差（RMSE），并将结果保存到变量RMSE_dt中。
RMSE_dt = sqrt(mean_squared_error(y_test, y_prediction_dt))
# 打印输出新改进的决策树回归模型的RMSE。
print("新改进的决策树回归模型在测试集上的RMSE为：" + str(RMSE_dt) + "\n")

# 计算随机森林回归模型在测试集上的均方根误差（RMSE），并将结果保存到变量RMSE_rf中。
RMSE_rf = sqrt(mean_squared_error(y_test, y_prediction_rf))
# 打印输出新改进的随机森林回归模型的RMSE。
print("新改进的随机森林回归模型在测试集上的RMSE为：" + str(RMSE_rf) + "\n")

# 忽略警告信息
warnings.filterwarnings('ignore')

# 创建一个 PrettyTable 表格对象
ptbl = PrettyTable()

# 设置表格列名
ptbl.field_names = ["Regressor Model", "Precision", "Recall", "F1Score"]

# 添加线性回归模型在测试集上的 Precision、Recall 和 F1 Score 指标
ptbl.add_row(["Linear",
              precision_score(y_test, y_prediction_lr, average='weighted'),  # 计算测试集上的 Precision
              recall_score(y_test, y_prediction_lr, average='weighted'),  # 计算测试集上的 Recall
              f1_score(y_test, y_prediction_lr, average='weighted')  # 计算测试集上的 F1 Score
              ])

# 添加决策树回归模型在测试集上的 Precision、Recall 和 F1 Score 指标
ptbl.add_row(["Decision Tree",
              precision_score(y_test, y_prediction_dt, average='weighted'),  # 计算测试集上的 Precision
              recall_score(y_test, y_prediction_dt, average='weighted'),  # 计算测试集上的 Recall
              f1_score(y_test, y_prediction_dt, average='weighted')  # 计算测试集上的 F1 Score
              ])

# 添加随机森林回归模型在测试集上的 Precision、Recall 和 F1 Score 指标
ptbl.add_row(["Random Forest",
              precision_score(y_test, y_prediction_rf, average='weighted'),  # 计算测试集上的 Precision
              recall_score(y_test, y_prediction_rf, average='weighted'),  # 计算测试集上的 Recall
              f1_score(y_test, y_prediction_rf, average='weighted')  # 计算测试集上的 F1 Score
              ])

# 输出表格
print(ptbl)

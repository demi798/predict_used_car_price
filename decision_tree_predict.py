import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("决策树模型 - 二手车价格预测")
print("=" * 80)

# 1. 读取数据
print("\n【第1步】读取数据...")
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"✓ 训练集: {train_df.shape[0]:,} 行, {train_df.shape[1]} 列")
print(f"✓ 测试集: {test_df.shape[0]:,} 行, {test_df.shape[1]} 列")

# 2. 数据预处理
print("\n【第2步】数据预处理...")

# 保存test集的SaleID用于最后输出
test_saleids = test_df['SaleID'].copy()

# 处理notRepairedDamage列：将'-'替换为NaN
train_df['notRepairedDamage'] = train_df['notRepairedDamage'].replace('-', np.nan)
test_df['notRepairedDamage'] = test_df['notRepairedDamage'].replace('-', np.nan)

# 转换为数值类型
train_df['notRepairedDamage'] = pd.to_numeric(train_df['notRepairedDamage'], errors='coerce')
test_df['notRepairedDamage'] = pd.to_numeric(test_df['notRepairedDamage'], errors='coerce')

# 3. 特征工程
print("【第3步】特征工程...")

# 创建车龄特征
train_df['regDate_str'] = train_df['regDate'].astype(str)
train_df['car_age'] = 2020 - train_df['regDate_str'].str[:4].astype(int)

test_df['regDate_str'] = test_df['regDate'].astype(str)
test_df['car_age'] = 2020 - test_df['regDate_str'].str[:4].astype(int)

# 4. 处理缺失值
print("【第4步】处理缺失值...")

# 选择需要的特征（去掉SaleID、name、regDate相关的原始列）
feature_cols = [col for col in train_df.columns 
                if col not in ['SaleID', 'name', 'regDate', 'creatDate', 'regDate_str', 'price']]

print(f"✓ 选择的特征数: {len(feature_cols)}")
print(f"  特征列表: {feature_cols}")

# 获取训练集特征
X_train = train_df[feature_cols].copy()
y_train = train_df['price'].copy()

# 获取测试集特征
X_test = test_df[feature_cols].copy()

# 缺失值填充 - 使用中位数
print("\n  处理缺失值...")
for col in feature_cols:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
        print(f"  ✓ {col}: 用中位数{median_val:.2f}填充")

# 检查是否还有NaN
if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    print("  警告：还有NaN值，使用0填充...")
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

print(f"\n✓ 训练集特征矩阵: {X_train.shape}")
print(f"✓ 测试集特征矩阵: {X_test.shape}")

# 5. 训练决策树模型
print("\n【第5步】训练决策树模型...")

# 使用决策树回归器
dt_model = DecisionTreeRegressor(
    max_depth=15,              # 限制树的深度，防止过拟合
    min_samples_split=10,       # 最少样本数以分割节点
    min_samples_leaf=5,         # 最少样本数以在叶子节点中
    random_state=42
)

print("  训练中...")
dt_model.fit(X_train, y_train)

# 评估训练集性能
train_score = dt_model.score(X_train, y_train)
print(f"\n✓ 训练集 R² 评分: {train_score:.4f}")

# 获取特征重要性
feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': dt_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\n✓ 特征重要性Top 10:")
print(feature_importance.head(10).to_string(index=False))

# 6. 在测试集上进行预测
print("\n【第6步】进行预测...")

y_pred = dt_model.predict(X_test)

print(f"✓ 预测完成")
print(f"  预测值统计:")
print(f"    - 最小值: {y_pred.min():.2f}")
print(f"    - 最大值: {y_pred.max():.2f}")
print(f"    - 平均值: {y_pred.mean():.2f}")
print(f"    - 中位数: {np.median(y_pred):.2f}")

# 7. 确保预测结果为整数且不为负数
y_pred = np.maximum(y_pred, 0).astype(int)

# 8. 生成提交结果
print("\n【第7步】生成提交文件...")

# 创建结果DataFrame
submission_df = pd.DataFrame({
    'SaleID': test_saleids,
    'price': y_pred
})

# 保存为CSV
output_file = 'prediction_result.csv'
submission_df.to_csv(output_file, index=False)

print(f"✓ 结果已保存: {output_file}")
print(f"\n  预览前10行:")
print(submission_df.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("预测完成！")
print("=" * 80)

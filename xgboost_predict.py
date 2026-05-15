import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("XGBoost模型 - 二手车价格预测（带验证集）")
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

# 获取训练集特征
X_train_full = train_df[feature_cols].copy()
y_train_full = train_df['price'].copy()

# 获取测试集特征
X_test = test_df[feature_cols].copy()

# 缺失值填充 - 使用中位数
print("  处理缺失值...")
for col in feature_cols:
    if X_train_full[col].isnull().sum() > 0:
        median_val = X_train_full[col].median()
        X_train_full[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# 检查是否还有NaN
if X_train_full.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    X_train_full.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

# 5. 划分训练集和验证集
print("\n【第5步】划分训练集和验证集...")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2,          # 20%用作验证集
    random_state=42
)

print(f"✓ 训练集: {X_train.shape[0]:,} 行")
print(f"✓ 验证集: {X_val.shape[0]:,} 行")
print(f"✓ 测试集: {X_test.shape[0]:,} 行")

# 6. 训练XGBoost模型
print("\n【第6步】训练XGBoost模型...")

# 创建DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

# XGBoost参数
params = {
    'objective': 'reg:squarederror',    # 回归任务
    'max_depth': 6,                      # 树的最大深度
    'learning_rate': 0.05,               # 学习率
    'subsample': 0.8,                    # 样本采样比
    'colsample_bytree': 0.8,             # 特征采样比
    'random_state': 42,
    'verbosity': 0                       # 不打印XGBoost内部日志
}

# 自定义评估函数：计算MAE
def mae_eval(y_pred, y_true):
    y_true_labels = y_true.get_label()
    mae = mean_absolute_error(y_true_labels, y_pred)
    return 'mae', mae

# 训练模型并监控验证集MAE
print("  训练中（监控验证集MAE）...\n")

evals_result = {}
evals = [(dtrain, 'train'), (dval, 'validation')]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,         # 迭代次数
    evals=evals,
    evals_result=evals_result,
    custom_metric=mae_eval,
    verbose_eval=10             # 每10轮打印一次评估结果
)

# 7. 模型评估
print("\n【第7步】模型评估...")

# 训练集预测和评估
y_train_pred = model.predict(dtrain)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# 验证集预测和评估
y_val_pred = model.predict(dval)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

print(f"\n✓ 训练集性能:")
print(f"  - MAE:  {train_mae:.2f}")
print(f"  - RMSE: {train_rmse:.2f}")
print(f"  - R²:   {train_r2:.4f}")

print(f"\n✓ 验证集性能:")
print(f"  - MAE:  {val_mae:.2f}")
print(f"  - RMSE: {val_rmse:.2f}")
print(f"  - R²:   {val_r2:.4f}")

# 8. 特征重要性
print("\n【第8步】特征重要性分析...")

feature_importance = model.get_score(importance_type='weight')
feature_importance_df = pd.DataFrame(list(feature_importance.items()), 
                                     columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)

print(f"\n✓ 特征重要性Top 10:")
print(feature_importance_df.head(10).to_string(index=False))

# 9. 在测试集上进行预测
print("\n【第9步】进行测试集预测...")

y_pred = model.predict(dtest)

print(f"✓ 预测完成")
print(f"  预测值统计:")
print(f"    - 最小值: {y_pred.min():.2f}")
print(f"    - 最大值: {y_pred.max():.2f}")
print(f"    - 平均值: {y_pred.mean():.2f}")
print(f"    - 中位数: {np.median(y_pred):.2f}")

# 10. 确保预测结果为整数且不为负数
y_pred = np.maximum(y_pred, 0).astype(int)

# 11. 生成提交结果
print("\n【第10步】生成提交文件...")

# 创建结果DataFrame
submission_df = pd.DataFrame({
    'SaleID': test_saleids,
    'price': y_pred
})

# 保存为CSV
output_file = 'xgboost_prediction_result.csv'
submission_df.to_csv(output_file, index=False)

print(f"✓ 结果已保存: {output_file}")
print(f"\n  预览前10行:")
print(submission_df.head(10).to_string(index=False))

# 12. 生成训练曲线
print("\n【第11步】生成训练曲线...")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
def setup_chinese_font():
    font_names = ['SimHei', 'STHeiti', 'PingFang SC', 'Heiti TC', 
                  'Songti SC', 'STSongti', 'Kaiti SC', 'Microsoft YaHei']
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font_name in font_names:
        if any(font_name in f for f in available_fonts):
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            return font_name
    
    plt.rcParams['font.sans-serif'] = fm.rcParams['font.sans-serif']
    return None

setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

# 绘制训练曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 迭代次数
iterations = range(len(evals_result['train']['rmse']))

# RMSE曲线
ax1.plot(iterations, evals_result['train']['rmse'], label='训练集', linewidth=2)
ax1.plot(iterations, evals_result['validation']['rmse'], label='验证集', linewidth=2)
ax1.set_xlabel('迭代次数')
ax1.set_ylabel('RMSE')
ax1.set_title('XGBoost训练过程 - RMSE变化')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 学习曲线
ax2.plot(iterations, evals_result['train']['mae'], label='训练集', linewidth=2)
ax2.plot(iterations, evals_result['validation']['mae'], label='验证集', linewidth=2)
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('MAE')
ax2.set_title('XGBoost训练过程 - MAE变化')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_training_curve.png', dpi=300, bbox_inches='tight')
print("✓ 训练曲线已保存: xgboost_training_curve.png")

print("\n" + "=" * 80)
print("XGBoost预测完成！")
print("=" * 80)

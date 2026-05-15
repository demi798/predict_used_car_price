import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("XGBoost模型优化版 - 二手车价格预测")
print("(更多轮数 + 小学习率 + 早停法 + 改进特征工程)")
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

# 3. 改进的特征工程
print("\n【第3步】改进的特征工程...")

print("  【特征工程优化说明】")
print("  1. 时间特征: 提取年份、月份、季度、车龄等")
print("  2. 交互特征: power*car_age、brand*bodyType等交叉特征")
print("  3. 统计特征: 对数变换、平方项等非线性特征")
print("  4. 分箱特征: 对连续特征进行分箱处理")

# 提取注册日期特征
for df in [train_df, test_df]:
    df['regDate_str'] = df['regDate'].astype(str).str.zfill(8)
    df['reg_year'] = df['regDate_str'].str[:4].astype(int)
    df['reg_month'] = df['regDate_str'].str[4:6].astype(int)
    df['car_age'] = 2020 - df['reg_year']
    df['reg_quarter'] = (df['reg_month'] - 1) // 3 + 1  # 季度
    
    # 提取上线时间特征
    df['creatDate_str'] = df['creatDate'].astype(str).str.zfill(8)
    df['create_year'] = df['creatDate_str'].str[:4].astype(int)
    df['create_month'] = df['creatDate_str'].str[4:6].astype(int)
    
    # 时间差：从注册到上线的时间差
    df['time_diff'] = df['creatDate'] - df['regDate']

# 4. 创建交互特征
print("\n  生成交互特征...")
for df in [train_df, test_df]:
    # 品牌与车体类型交互
    df['brand_body'] = df['brand'].astype(str) + '_' + df['bodyType'].astype(str)
    
    # 功率与车龄的交互
    df['power_age'] = df['power'] * df['car_age']
    
    # 里程与车龄的交互
    df['km_age'] = df['kilometer'] * df['car_age']
    
    # 功率的对数
    df['log_power'] = np.log1p(df['power'])
    
    # 里程的对数
    df['log_km'] = np.log1p(df['kilometer'])
    
    # 功率的平方
    df['power_sq'] = df['power'] ** 2
    
    # 里程的分箱
    df['km_bin'] = pd.cut(df['kilometer'], bins=[0, 5, 10, 15], labels=[0, 1, 2], include_lowest=True)

# 5. 处理缺失值
print("\n【第4步】处理缺失值...")

# 选择需要的特征
feature_cols = [col for col in train_df.columns 
                if col not in ['SaleID', 'name', 'regDate', 'creatDate', 'regDate_str', 'creatDate_str', 
                               'price', 'reg_year', 'create_year', 'reg_month', 'create_month']]

print(f"✓ 选择的特征数: {len(feature_cols)}")

# 获取训练集特征
X_train_full = train_df[feature_cols].copy()
y_train_full = train_df['price'].copy()

# 获取测试集特征
X_test = test_df[feature_cols].copy()

# 处理分类列brand_body
if 'brand_body' in X_train_full.columns:
    # 创建映射字典，未见过的值映射为-1
    brand_body_map = {val: idx for idx, val in enumerate(X_train_full['brand_body'].unique())}
    X_train_full['brand_body'] = X_train_full['brand_body'].astype(str).map(brand_body_map)
    X_test['brand_body'] = X_test['brand_body'].astype(str).map(lambda x: brand_body_map.get(x, -1))

# 处理km_bin
if 'km_bin' in X_train_full.columns:
    X_train_full['km_bin'] = X_train_full['km_bin'].astype(int)
    X_test['km_bin'] = X_test['km_bin'].astype(int)

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

# 6. 划分训练集和验证集
print("\n【第5步】划分训练集和验证集...")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2,
    random_state=42
)

print(f"✓ 训练集: {X_train.shape[0]:,} 行, {X_train.shape[1]} 列")
print(f"✓ 验证集: {X_val.shape[0]:,} 行")
print(f"✓ 测试集: {X_test.shape[0]:,} 行")

# 7. 训练XGBoost模型（优化配置）
print("\n【第6步】训练XGBoost模型（优化配置）...")

# 创建DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

# XGBoost参数 - 优化版
params = {
    'objective': 'reg:squarederror',
    'max_depth': 7,                    # 略微加深
    'learning_rate': 0.01,             # 更小的学习率
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,          # 额外的特征采样
    'random_state': 42,
    'verbosity': 0,
    'lambda': 1.0,                     # L2正则化
    'alpha': 0.0                       # L1正则化
}

print("  参数配置:")
print(f"    - learning_rate: 0.01（更小的学习率）")
print(f"    - max_depth: 7")
print(f"    - max_boost_rounds: 500")
print(f"    - early_stopping_rounds: 20")

# 训练模型，使用早停法
print("\n  训练中（启用早停法）...\n")

evals_result = {}
evals = [(dtrain, 'train'), (dval, 'validation')]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,               # 更多轮数
    evals=evals,
    evals_result=evals_result,
    early_stopping_rounds=20,          # 早停法：如果验证集MAE 20轮内没有改进则停止
    verbose_eval=20                    # 每20轮打印一次
)

best_iteration = model.best_iteration
print(f"\n✓ 早停法触发: 第{best_iteration}轮停止（早停轮数: 20）")

# 8. 模型评估
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
print(f"  - MAE:  {val_mae:.2f}  ⭐")
print(f"  - RMSE: {val_rmse:.2f}")
print(f"  - R²:   {val_r2:.4f}")

# 9. 特征重要性
print("\n【第8步】特征重要性分析...")

feature_importance = model.get_score(importance_type='weight')
feature_importance_df = pd.DataFrame(list(feature_importance.items()), 
                                     columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)

print(f"\n✓ 特征重要性Top 15:")
print(feature_importance_df.head(15).to_string(index=False))

# 10. 在测试集上进行预测
print("\n【第9步】进行测试集预测...")

y_pred = model.predict(dtest)

print(f"✓ 预测完成")
print(f"  预测值统计:")
print(f"    - 最小值: {y_pred.min():.2f}")
print(f"    - 最大值: {y_pred.max():.2f}")
print(f"    - 平均值: {y_pred.mean():.2f}")
print(f"    - 中位数: {np.median(y_pred):.2f}")

# 确保预测结果为整数且不为负数
y_pred = np.maximum(y_pred, 0).astype(int)

# 11. 生成提交结果
print("\n【第10步】生成提交文件...")

submission_df = pd.DataFrame({
    'SaleID': test_saleids,
    'price': y_pred
})

output_file = 'xgboost_optimized_result.csv'
submission_df.to_csv(output_file, index=False)

print(f"✓ 结果已保存: {output_file}")
print(f"\n  预览前10行:")
print(submission_df.head(10).to_string(index=False))

# 12. 生成训练曲线
print("\n【第11步】生成训练曲线...")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

iterations = range(len(evals_result['train']['rmse']))

# RMSE曲线
ax1.plot(iterations, evals_result['train']['rmse'], label='训练集', linewidth=2)
ax1.plot(iterations, evals_result['validation']['rmse'], label='验证集', linewidth=2)
ax1.axvline(best_iteration, color='red', linestyle='--', label=f'早停点 ({best_iteration})', linewidth=2)
ax1.set_xlabel('迭代次数')
ax1.set_ylabel('RMSE')
ax1.set_title('XGBoost优化模型 - RMSE变化（启用早停法）')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 计算MAE用于展示
train_rmse_list = evals_result['train']['rmse']
val_rmse_list = evals_result['validation']['rmse']

# RMSE和MAE的关系：对于正态分布，MAE ≈ 0.8*RMSE
train_mae_est = [r * 0.55 for r in train_rmse_list]  # 估计的MAE
val_mae_est = [r * 0.55 for r in val_rmse_list]

ax2.plot(iterations, train_mae_est, label='训练集(估计)', linewidth=2)
ax2.plot(iterations, val_mae_est, label='验证集(估计)', linewidth=2)
ax2.axvline(best_iteration, color='red', linestyle='--', label=f'早停点 ({best_iteration})', linewidth=2)
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('MAE (估计值)')
ax2.set_title('XGBoost优化模型 - MAE变化估计')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_optimized_training_curve.png', dpi=300, bbox_inches='tight')
print("✓ 训练曲线已保存: xgboost_optimized_training_curve.png")

# 13. 特征工程总结
print("\n【第12步】特征工程优化总结...")
print("\n✓ 已实现的特征工程优化:")
print("  1. 时间特征:")
print("     - 提取年份、月份、季度")
print("     - 计算车龄（2020 - 注册年份）")
print("     - 上线时间的年月信息")
print("     - 时间差特征（上线时间 - 注册时间）")
print("\n  2. 交互特征:")
print("     - power * car_age (功率与车龄)")
print("     - kilometer * car_age (里程与车龄)")
print("     - brand_body (品牌与车体类型编码)")
print("\n  3. 非线性特征:")
print("     - log_power (功率对数变换)")
print("     - log_km (里程对数变换)")
print("     - power_sq (功率平方)")
print("\n  4. 分箱特征:")
print("     - km_bin (里程分箱：0-5, 5-10, 10-15万)")

print("\n✓ 后续可优化方向:")
print("  1. 特征选择：使用特征选择算法移除低重要性特征")
print("  2. 更多交互特征：尝试更多特征组合")
print("  3. 目标变量变换：对price进行log变换（因为右偏）")
print("  4. 类别特征编码：尝试不同的编码方式（target encoding等）")
print("  5. 超参数调优：使用贝叶斯优化搜索最优参数")
print("  6. 集成学习：结合多个模型的预测结果")
print("  7. 数据清洗：处理异常值和离群点")
print("  8. 特征缩放：对某些特征进行标准化处理")

print("\n" + "=" * 80)
print("XGBoost优化模型训练完成！")
print("=" * 80)

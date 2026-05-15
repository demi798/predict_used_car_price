import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("CatBoost模型 - 高级特征工程版 - 二手车价格预测")
print("包含：品牌等级、目标编码、统计特征、异常标志等")
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

# 3. 高级特征工程
print("\n【第3步】高级特征工程...")

print("  【特征工程方案】")
print("  ┌─ 时间特征：车龄分段、年均里程、注册季节")
print("  ├─ 品牌特征：品牌分级、品牌均价、目标编码")
print("  ├─ 车况特征：里程分段、功率分级、性价比")
print("  ├─ 异常标志：缺失值标记、异常里程标记")
print("  └─ 交互特征：功率-车龄、里程-车龄")

# ========== 3.1 时间特征 ==========
print("\n  [1/5] 时间特征提取...")
for df in [train_df, test_df]:
    df['regDate_str'] = df['regDate'].astype(str).str.zfill(8)
    df['reg_year'] = df['regDate_str'].str[:4].astype(int)
    df['reg_month'] = df['regDate_str'].str[4:6].astype(int)
    df['car_age'] = 2020 - df['reg_year']
    
    # 注册季度（季节性）
    df['reg_quarter'] = (df['reg_month'] - 1) // 3 + 1
    df['reg_season'] = df['reg_quarter'].map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'})
    
    # 上线时间特征
    df['creatDate_str'] = df['creatDate'].astype(str).str.zfill(8)
    df['create_year'] = df['creatDate_str'].str[:4].astype(int)
    df['create_month'] = df['creatDate_str'].str[4:6].astype(int)
    
    # 时间差：从注册到上线的延迟（天数）
    df['time_diff'] = df['creatDate'] - df['regDate']
    
    # 车龄分段（👉 第一优先级）
    df['car_age_segment'] = pd.cut(df['car_age'], 
                                   bins=[0, 3, 5, 10, 100], 
                                   labels=['0-3年', '3-5年', '5-10年', '10+年'],
                                   include_lowest=True)
    
    # 注册年份分组
    df['reg_year_period'] = pd.cut(df['reg_year'],
                                   bins=[1990, 2005, 2010, 2015, 2020],
                                   labels=['2005前', '2005-2010', '2010-2015', '2015-2020'],
                                   include_lowest=True)

# ========== 3.2 品牌特征 ==========
print("  [2/5] 品牌特征工程...")

# 计算品牌的统计信息（在训练集上计算，用于编码）
brand_stats = train_df.groupby('brand')['price'].agg(['mean', 'std', 'count']).reset_index()
brand_stats.columns = ['brand', 'brand_avg_price', 'brand_price_std', 'brand_count']

# 品牌分级：根据平均价格
brand_stats['brand_tier'] = pd.qcut(brand_stats['brand_avg_price'], 
                                     q=4, 
                                     labels=['budget', 'economy', 'premium', 'luxury'],
                                     duplicates='drop')

# 将品牌统计信息合并到原数据
train_df = train_df.merge(brand_stats[['brand', 'brand_avg_price', 'brand_price_std', 'brand_count', 'brand_tier']], 
                          on='brand', how='left')
test_df = test_df.merge(brand_stats[['brand', 'brand_avg_price', 'brand_price_std', 'brand_count', 'brand_tier']], 
                        on='brand', how='left')

# ========== 3.3 车况特征 ==========
print("  [3/5] 车况特征工程...")

for df in [train_df, test_df]:
    # 年均里程（👉 第一优先级）- 捕捉车况质量
    df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)  # +1避免除以0
    
    # 里程分段
    df['km_segment'] = pd.cut(df['kilometer'],
                              bins=[0, 5, 10, 15, 20, 1000],
                              labels=['0-5万', '5-10万', '10-15万', '15-20万', '20+万'],
                              include_lowest=True)
    
    # 功率分级
    df['power_segment'] = pd.cut(df['power'],
                                bins=[0, 100, 150, 200, 5000],
                                labels=['低功率', '中功率', '高功率', '超高功率'],
                                include_lowest=True)
    
    # 性价比特征：功率/里程（同等车况下，功率越高越贵）
    df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
    
    # 功率/价格倾向的代理（使用品牌均价作为参考）
    df['power_to_brand_price'] = df['power'] / (df['brand_avg_price'] + 1)

# ========== 3.4 异常标志特征 ==========
print("  [4/5] 异常标志特征...")

for df in [train_df, test_df]:
    # 缺失值标记
    df['notRepaired_is_missing'] = df['notRepairedDamage'].isna().astype(int)
    
    # 异常里程标记：年均里程过高（可能事故车被高使用）
    df['km_per_year_high'] = (df['km_per_year'] > 2.5).astype(int)  # 平均年均里程约2.5万
    
    # 新车却里程很高（可能事故车或营运车）
    df['new_car_high_km'] = ((df['car_age'] <= 2) & (df['kilometer'] > 10)).astype(int)
    
    # 异常功率标记：功率为0
    df['power_is_zero'] = (df['power'] == 0).astype(int)
    
    # 异常时间差：注册后很长时间才上线（可能长期库存）
    df['long_listing_delay'] = (df['time_diff'] > 365).astype(int)

# ========== 3.5 交互特征 ==========
print("  [5/5] 交互特征...")

for df in [train_df, test_df]:
    # 功率-车龄交互
    df['power_age'] = df['power'] * df['car_age']
    
    # 里程-车龄交互
    df['km_age'] = df['kilometer'] * df['car_age']
    
    # 高性能-年轻车型交互（豪车倾向）
    df['high_power_young'] = ((df['power'] > 200) & (df['car_age'] <= 3)).astype(int)
    
    # 低里程-年轻车型交互（准新车）
    df['low_km_young'] = ((df['kilometer'] <= 5) & (df['car_age'] <= 2)).astype(int)
    
    # 对数变换（处理偏态分布）
    df['log_power'] = np.log1p(df['power'])
    df['log_km'] = np.log1p(df['kilometer'])
    df['log_brand_avg_price'] = np.log1p(df['brand_avg_price'])

# ========== 3.6 分类特征编码（目标编码） ==========
print("\n  [编码] 分类特征编码（目标编码）...")

# 定义需要目标编码的特征
categorical_features_to_encode = ['bodyType', 'fuelType', 'gearbox', 'regionCode']

target_encoders = {}

for feature in categorical_features_to_encode:
    # 基于训练集计算目标编码
    target_encoding = train_df.groupby(feature)['price'].mean().to_dict()
    target_encoders[feature] = target_encoding
    
    # 应用到训练集和测试集
    # 对于未见过的值，使用全局平均值
    global_mean = train_df['price'].mean()
    train_df[f'{feature}_encoded'] = train_df[feature].map(target_encoding).fillna(global_mean)
    test_df[f'{feature}_encoded'] = test_df[feature].map(target_encoding).fillna(global_mean)

# 品牌分级编码
brand_tier_encoding = {'budget': 0, 'economy': 1, 'premium': 2, 'luxury': 3}
train_df['brand_tier_encoded'] = train_df['brand_tier'].map(brand_tier_encoding)
test_df['brand_tier_encoded'] = test_df['brand_tier'].map(brand_tier_encoding)

# 车龄分段编码
age_segment_encoding = {'0-3年': 0, '3-5年': 1, '5-10年': 2, '10+年': 3}
train_df['car_age_segment_encoded'] = train_df['car_age_segment'].map(age_segment_encoding)
test_df['car_age_segment_encoded'] = test_df['car_age_segment'].map(age_segment_encoding)

# 里程分段编码
km_segment_encoding = {'0-5万': 0, '5-10万': 1, '10-15万': 2, '15-20万': 3, '20+万': 4}
train_df['km_segment_encoded'] = train_df['km_segment'].map(km_segment_encoding)
test_df['km_segment_encoded'] = test_df['km_segment'].map(km_segment_encoding)

# 功率分段编码
power_segment_encoding = {'低功率': 0, '中功率': 1, '高功率': 2, '超高功率': 3}
train_df['power_segment_encoded'] = train_df['power_segment'].map(power_segment_encoding)
test_df['power_segment_encoded'] = test_df['power_segment'].map(power_segment_encoding)

# 注册季节编码
season_encoding = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
train_df['reg_season_encoded'] = train_df['reg_season'].map(season_encoding)
test_df['reg_season_encoded'] = test_df['reg_season'].map(season_encoding)

# ========== 3.7 无预测价值特征分析 ==========
print("  [清理] 分析无预测价值特征（仅删除显式无关字段）...")

# 只删除明确无关的元数据字段，避免误删可能有价值的特征
drop_features = ['SaleID', 'name', 'offerType', 'seller', 'regDate', 'creatDate', 'regDate_str', 'creatDate_str']
train_df = train_df.drop(columns=drop_features, errors='ignore')
test_df = test_df.drop(columns=drop_features, errors='ignore')

# 检查常数特征（对模型无预测价值），如果存在则也删除
constant_features = [col for col in train_df.columns if col != 'price' and train_df[col].nunique(dropna=False) <= 1]
if constant_features:
    print(f"  ✓ 常数字段（无预测价值）: {constant_features}")
    train_df = train_df.drop(columns=constant_features, errors='ignore')
    test_df = test_df.drop(columns=constant_features, errors='ignore')
else:
    print("  ✓ 未发现常数字段")

# 计算数值特征与目标相关性，以帮助识别候选低价值特征，但不自动删除它们
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop('price', errors='ignore')
if len(numeric_cols) > 0:
    corr_with_target = train_df[numeric_cols].corrwith(train_df['price']).abs().sort_values()
    low_corr = corr_with_target[corr_with_target < 0.01]
    if not low_corr.empty:
        print(f"  ⚠ 低相关特征候选（仅作参考，不自动删除）: {low_corr.index.tolist()}")
    else:
        print("  ✓ 未发现显著低相关数值特征")

# 4. 特征选择
print("\n【第4步】特征选择...")

# 排除目标变量和非特征列
feature_cols = [col for col in train_df.columns if col != 'price']

print(f"✓ 特征总数: {len(feature_cols)}")
print(f"\n特征列表:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# 获取特征和标签
X_train_full = train_df[feature_cols].copy()
y_train_full = train_df['price'].copy()
X_test = test_df[feature_cols].copy()

# 定义 CatBoost 的原始类别特征
cat_features = [
    'brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'regionCode',
    'brand_tier', 'car_age_segment', 'km_segment', 'power_segment',
    'reg_season', 'reg_year_period'
]
cat_features = [col for col in cat_features if col in X_train_full.columns]

# 自动补充所有 pandas category 列到 cat_features，避免遗漏
additional_cat_features = [
    col for col in X_train_full.select_dtypes(include=['category']).columns
    if col not in cat_features
]
if additional_cat_features:
    cat_features += additional_cat_features

# ========== 4.1 处理缺失值 ==========
print("\n【第5步】处理缺失值...")

# 检查缺失值
missing_info = X_train_full.isnull().sum()
if missing_info.sum() > 0:
    print("  发现缺失值:")
    print(missing_info[missing_info > 0])
else:
    print("  ✓ 无缺失值")

# 缺失值填充 - 对 cat_features 使用字符串缺失值，其它数值列使用中位数
for col in feature_cols:
    if X_train_full[col].isnull().sum() > 0:
        if col in cat_features:
            X_train_full[col] = X_train_full[col].astype(object).fillna('missing')
            X_test[col] = X_test[col].astype(object).fillna('missing')
        elif X_train_full[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            median_val = X_train_full[col].median()
            X_train_full[col] = X_train_full[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
        else:
            X_train_full[col] = X_train_full[col].astype(object).fillna('missing')
            X_test[col] = X_test[col].astype(object).fillna('missing')

# 确保没有NaN
if X_train_full.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    for col in X_train_full.columns:
        if col in cat_features:
            X_train_full[col] = X_train_full[col].astype(object).fillna('missing')
            X_test[col] = X_test[col].astype(object).fillna('missing')
        elif X_train_full[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            X_train_full[col] = X_train_full[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)
        else:
            X_train_full[col] = X_train_full[col].astype(object).fillna('missing')
            X_test[col] = X_test[col].astype(object).fillna('missing')

# 将 CatBoost 类别特征统一为字符串，避免浮点类别报错
for col in cat_features:
    if col in X_train_full.columns:
        X_train_full[col] = X_train_full[col].astype(str)
        X_test[col] = X_test[col].astype(str)

# 如果还有 pandas category dtype 列，则也统一转字符串并纳入 cat_features
remaining_category_cols = [
    col for col in X_train_full.select_dtypes(include=['category']).columns
    if col not in cat_features
]
for col in remaining_category_cols:
    cat_features.append(col)
    X_train_full[col] = X_train_full[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# CatBoost 将直接使用原始类别特征，不需要手工编码对象列

# 5. 划分训练集和验证集
print("\n【第6步】划分训练集和验证集...")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42
)

print(f"✓ 训练集: {X_train.shape[0]:,} 行, {X_train.shape[1]} 列")
print(f"✓ 验证集: {X_val.shape[0]:,} 行")
print(f"✓ 测试集: {X_test.shape[0]:,} 行")

# 6. 训练CatBoost模型
print("\n【第7步】训练CatBoost模型...")

print("  参数配置:")
print(f"    - learning_rate: 0.01")
print(f"    - depth: 7")
print(f"    - iterations: 10000")
print(f"    - early_stopping_rounds: 20")

print("\n  训练中（启用早停法）...\n")

model = CatBoostRegressor(
    iterations=10000,
    learning_rate=0.01,
    depth=7,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    od_type='Iter',
    od_wait=20,
    verbose=20,
    allow_writing_files=False
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    use_best_model=True,
    verbose=20
)

best_iteration = model.get_best_iteration()
print(f"\n✓ 早停法触发: 第{best_iteration}轮停止")

# 7. 模型评估
print("\n【第8步】模型评估...")

# 训练集评估
y_train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# 验证集评估
y_val_pred = model.predict(X_val)
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

# 8. 特征重要性
print("\n【第9步】特征重要性分析...")

feature_importance = model.get_feature_importance(type='FeatureImportance')
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"\n✓ 特征重要性 Top 20:")
print(feature_importance_df.head(20).to_string(index=False))

# 9. 测试集预测
print("\n【第10步】进行测试集预测...")

y_pred = model.predict(X_test)

print(f"✓ 预测完成")
print(f"  预测值统计:")
print(f"    - 最小值: {y_pred.min():.2f}")
print(f"    - 最大值: {y_pred.max():.2f}")
print(f"    - 平均值: {y_pred.mean():.2f}")
print(f"    - 中位数: {np.median(y_pred):.2f}")

# 确保预测结果为整数且不为负数
y_pred = np.maximum(y_pred, 0).astype(int)

# 10. 生成提交结果
print("\n【第11步】生成提交文件...")

submission_df = pd.DataFrame({
    'SaleID': test_saleids,
    'price': y_pred
})

output_file = 'catboost_advanced_features_result_new.csv'
submission_df.to_csv(output_file, index=False)

print(f"✓ 结果已保存: {output_file}")
print(f"\n  预览前10行:")
print(submission_df.head(10).to_string(index=False))

# 11. 生成训练曲线
print("\n【第12步】生成训练曲线...")

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

evals_result = model.get_evals_result()
train_rmse_list = evals_result['learn']['RMSE']
val_rmse_list = evals_result['validation']['RMSE']
iterations = range(len(train_rmse_list))

# RMSE曲线
ax1.plot(iterations, train_rmse_list, label='训练集', linewidth=2)
ax1.plot(iterations, val_rmse_list, label='验证集', linewidth=2)
ax1.axvline(best_iteration, color='red', linestyle='--', label=f'早停点 ({best_iteration})', linewidth=2)
ax1.set_xlabel('迭代次数')
ax1.set_ylabel('RMSE')
ax1.set_title('CatBoost高级特征版 - RMSE变化')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MAE估计曲线
train_mae_est = [r * 0.55 for r in train_rmse_list]
val_mae_est = [r * 0.55 for r in val_rmse_list]

ax2.plot(iterations, train_mae_est, label='训练集(估计)', linewidth=2)
ax2.plot(iterations, val_mae_est, label='验证集(估计)', linewidth=2)
ax2.axvline(best_iteration, color='red', linestyle='--', label=f'早停点 ({best_iteration})', linewidth=2)
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('MAE (估计值)')
ax2.set_title('CatBoost高级特征版 - MAE变化')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('catboost_advanced_features_training_curve.png', dpi=300, bbox_inches='tight')
print("✓ 训练曲线已保存: catboost_advanced_features_training_curve.png")

# 12. 特征工程总结
print("\n【第13步】高级特征工程总结...")

print("\n✓ 已实现的特征工程（共5大类）:")
print("\n  【1】时间特征（3个新特征）:")
print("     ✓ car_age_segment: 车龄分段（0-3年、3-5年等）")
print("     ✓ km_per_year: 年均里程（反映车况质量）")
print("     ✓ reg_season_encoded: 注册季节编码（季节性影响）")
print("     ✓ reg_year_period: 注册年份分组")
print("     ✓ time_diff: 注册到上线延迟")

print("\n  【2】品牌特征（4个新特征）:")
print("     ✓ brand_avg_price: 品牌平均价格（市场定位）")
print("     ✓ brand_price_std: 品牌价格标准差（差异度）")
print("     ✓ brand_count: 品牌数量（市场热度）")
print("     ✓ brand_tier_encoded: 品牌分级（豪车/经济/预算）")

print("\n  【3】车况特征（5个新特征）:")
print("     ✓ km_segment_encoded: 里程分段（车况等级）")
print("     ✓ power_segment_encoded: 功率分级（性能等级）")
print("     ✓ power_km_ratio: 性价比特征")
print("     ✓ power_to_brand_price: 功率相对品牌价格")
print("     ✓ log_km, log_power: 对数变换（处理偏态）")

print("\n  【4】异常标志特征（5个新特征）:")
print("     ✓ notRepaired_is_missing: 缺失值标记")
print("     ✓ km_per_year_high: 年均里程过高标记")
print("     ✓ new_car_high_km: 新车高里程异常标记")
print("     ✓ power_is_zero: 功率为0标记")
print("     ✓ long_listing_delay: 长期库存标记")

print("\n  【5】交互特征（4个新特征）:")
print("     ✓ power_age: 功率×车龄")
print("     ✓ km_age: 里程×车龄")
print("     ✓ high_power_young: 高性能年轻车")
print("     ✓ low_km_young: 低里程年轻车")

print("\n  【6】分类特征编码:")
print("     ✓ bodyType_encoded: 车体类型（目标编码）")
print("     ✓ fuelType_encoded: 燃油类型（目标编码）")
print("     ✓ gearbox_encoded: 变速箱类型（目标编码）")
print("     ✓ regionCode_encoded: 地区代码（目标编码）")

print("\n  【7】删除的无用特征:")
print("     ✗ SaleID, name, offerType, seller")
print("     ✗ 原始日期和中间变量")

print("\n✓ 特征工程收益:")
print(f"  - 特征数量: 31 → {len(feature_cols)}")
print(f"  - 验证集MAE: 基准线 vs {val_mae:.2f}")
print(f"  - 验证集RMSE: 基准线 vs {val_rmse:.2f}")
print(f"  - 验证集R²: {val_r2:.4f}")

print("\n" + "=" * 80)
print("CatBoost高级特征版训练完成！")
print("=" * 80)

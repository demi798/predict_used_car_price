import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("TensorFlow二手车价格预测脚本（特征工程增强版）")

# 1. 读取数据
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

test_saleids = test_df['SaleID'].copy()

# 2. 预处理日期 and 缺失值
for df in [train_df, test_df]:
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
    df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')
    df['regDate_str'] = df['regDate'].astype(str).str.zfill(8)
    df['creatDate_str'] = df['creatDate'].astype(str).str.zfill(8)
    df['reg_year'] = df['regDate_str'].str[:4].astype(int)
    df['reg_month'] = df['regDate_str'].str[4:6].astype(int)
    df['create_year'] = df['creatDate_str'].str[:4].astype(int)
    df['create_month'] = df['creatDate_str'].str[4:6].astype(int)
    df['car_age'] = 2020 - df['reg_year']
    df['reg_quarter'] = ((df['reg_month'] - 1) // 3 + 1).clip(1, 4)
    df['reg_season'] = df['reg_quarter'].map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'})
    df['time_diff'] = df['creatDate'] - df['regDate']
    df['car_age_segment'] = pd.cut(df['car_age'], bins=[0, 3, 5, 10, 100],
                                   labels=['0-3年', '3-5年', '5-10年', '10+年'], include_lowest=True)
    df['reg_year_period'] = pd.cut(df['reg_year'], bins=[1990, 2005, 2010, 2015, 2020],
                                   labels=['2005前', '2005-2010', '2010-2015', '2015-2020'], include_lowest=True)

# 3. 品牌特征统计信息
brand_stats = train_df.groupby('brand')['price'].agg(['mean', 'std', 'count']).reset_index()
brand_stats.columns = ['brand', 'brand_avg_price', 'brand_price_std', 'brand_count']
brand_stats['brand_tier'] = pd.qcut(brand_stats['brand_avg_price'], q=4,
                                    labels=['budget', 'economy', 'premium', 'luxury'], duplicates='drop')

train_df = train_df.merge(brand_stats, on='brand', how='left')
test_df = test_df.merge(brand_stats, on='brand', how='left')

# 4. 车况与交互特征
for df in [train_df, test_df]:
    df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)
    df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
    df['power_to_brand_price'] = df['power'] / (df['brand_avg_price'] + 1)
    df['power_age'] = df['power'] * df['car_age']
    df['km_age'] = df['kilometer'] * df['car_age']
    df['high_power_young'] = ((df['power'] > 200) & (df['car_age'] <= 3)).astype(int)
    df['low_km_young'] = ((df['kilometer'] <= 5) & (df['car_age'] <= 2)).astype(int)
    df['log_kilometer'] = np.log1p(df['kilometer'])
    df['log_power'] = np.log1p(df['power'])
    df['log_brand_avg_price'] = np.log1p(df['brand_avg_price'])
    df['notRepaired_is_missing'] = df['notRepairedDamage'].isna().astype(int)
    df['km_per_year_high'] = (df['km_per_year'] > 2.5).astype(int)
    df['new_car_high_km'] = ((df['car_age'] <= 2) & (df['kilometer'] > 10)).astype(int)
    df['power_is_zero'] = (df['power'] == 0).astype(int)
    df['long_listing_delay'] = (df['time_diff'] > 365).astype(int)

# 5. 目标编码分类变量
encoding_features = ['bodyType', 'fuelType', 'gearbox', 'regionCode', 'brand']
global_mean = train_df['price'].mean()
for feature in encoding_features:
    encoding_map = train_df.groupby(feature)['price'].mean().to_dict()
    train_df[f'{feature}_price_mean'] = train_df[feature].map(encoding_map).fillna(global_mean)
    test_df[f'{feature}_price_mean'] = test_df[feature].map(encoding_map).fillna(global_mean)

# 6. 类别分段编码
car_age_segment_map = {'0-3年': 0, '3-5年': 1, '5-10年': 2, '10+年': 3}
km_segment_map = {'0-5万': 0, '5-10万': 1, '10-15万': 2, '15-20万': 3, '20+万': 4}
power_segment_map = {'低功率': 0, '中功率': 1, '高功率': 2, '超高功率': 3}
reg_season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
reg_year_period_map = {'2005前': 0, '2005-2010': 1, '2010-2015': 2, '2015-2020': 3}

for df in [train_df, test_df]:
    df['car_age_segment_encoded'] = df['car_age_segment'].map(car_age_segment_map).astype(float).fillna(-1)
    df['reg_season_encoded'] = df['reg_season'].map(reg_season_map).astype(float).fillna(-1)
    df['reg_year_period_encoded'] = df['reg_year_period'].map(reg_year_period_map).astype(float).fillna(-1)
    df['km_segment'] = pd.cut(df['kilometer'], bins=[0, 5, 10, 15, 20, 1000],
                               labels=['0-5万', '5-10万', '10-15万', '15-20万', '20+万'], include_lowest=True)
    df['km_segment_encoded'] = df['km_segment'].map(km_segment_map).astype(float).fillna(-1)
    df['power_segment'] = pd.cut(df['power'], bins=[0, 100, 150, 200, 5000],
                                 labels=['低功率', '中功率', '高功率', '超高功率'], include_lowest=True)
    df['power_segment_encoded'] = df['power_segment'].map(power_segment_map).astype(float).fillna(-1)

# 7. 删除无关字段
drop_cols = ['SaleID', 'name', 'offerType', 'seller', 'regDate', 'creatDate',
             'regDate_str', 'creatDate_str', 'reg_quarter', 'reg_season', 'car_age_segment',
             'reg_year_period', 'km_segment', 'power_segment']
train_df = train_df.drop(columns=drop_cols, errors='ignore')
test_df = test_df.drop(columns=drop_cols, errors='ignore')

# 8. 特征列表准备
exclude_columns = ['price']
feature_cols = [col for col in train_df.columns if col not in exclude_columns]

print(f"✓ 训练集特征数量: {len(feature_cols)}")
print("✓ 部分特征:", feature_cols[:20])

X = train_df[feature_cols].copy()
y = train_df['price'].astype(float).copy()
X_test = test_df[feature_cols].copy()

# 9. 缺失值填充
for col in X.columns:
    if X[col].isnull().any() or X_test[col].isnull().any():
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype(object).fillna('missing')
            X_test[col] = X_test[col].astype(object).fillna('missing')
        else:
            median_value = X[col].median()
            X[col] = X[col].fillna(median_value)
            X_test[col] = X_test[col].fillna(median_value)

# 10. 确保数值型输入
for col in X.select_dtypes(include=['object', 'category', 'string']).columns:
    categories = pd.Categorical(X[col].astype(str)).categories
    X[col] = pd.Categorical(X[col].astype(str), categories=categories).codes.astype(float)
    X_test[col] = pd.Categorical(X_test[col].astype(str), categories=categories).codes.astype(float)

X = X.astype(float)
X_test = X_test.astype(float)

# 11. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 12. 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"✓ 训练集: {X_train.shape[0]:,} 行")
print(f"✓ 验证集: {X_val.shape[0]:,} 行")

# 13. 模型结构
inputs = tf.keras.Input(shape=(X_train.shape[1],), name='input_features')
x = inputs
for i in range(16):
    x = tf.keras.layers.Dense(128, activation='relu', name=f'dense_{i + 1}')(x)
    x = tf.keras.layers.BatchNormalization(name=f'bn_{i + 1}')(x)
    x = tf.keras.layers.Dropout(0.1, name=f'dropout_{i + 1}')(x)
outputs = tf.keras.layers.Dense(1, activation='linear', name='price_output')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='tf_price_model')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error',
    metrics=['mean_absolute_error', 'mean_squared_error']
)

model.summary()

# 14. 训练
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=512,
    callbacks=callbacks,
    verbose=2
)

# 15. 评估与预测
val_loss, val_mae, val_mse = model.evaluate(X_val, y_val, verbose=0)
print(f"验证集结果: MAE={val_mae:.2f}, MSE={val_mse:.2f}")

preds = model.predict(X_test_scaled, batch_size=512).flatten()
preds = np.maximum(preds, 0).astype(int)

submission = pd.DataFrame({'SaleID': test_saleids, 'price': preds})
output_file = 'tensorflow_price_prediction_refined.csv'
submission.to_csv(output_file, index=False)

print(f"✓ 预测文件已生成: {output_file}")
print(submission.head(10).to_string(index=False))

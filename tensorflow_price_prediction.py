import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("TensorFlow二手车价格预测脚本")

# 读取数据
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 目标变量
train_df['notRepairedDamage'] = train_df['notRepairedDamage'].replace('-', np.nan)
test_df['notRepairedDamage'] = test_df['notRepairedDamage'].replace('-', np.nan)
train_df['notRepairedDamage'] = pd.to_numeric(train_df['notRepairedDamage'], errors='coerce').fillna(-1).astype(float)
test_df['notRepairedDamage'] = pd.to_numeric(test_df['notRepairedDamage'], errors='coerce').fillna(-1).astype(float)

# 特征选择
exclude_columns = ['SaleID', 'name', 'price']
feature_cols = [col for col in train_df.columns if col not in exclude_columns]

# 准备特征
X_train = train_df[feature_cols].copy()
y_train = train_df['price'].astype(float).copy()
X_test = test_df[feature_cols].copy()

# 用训练集统计值填充测试集可能存在的缺失值
for col in X_train.columns:
    if X_train[col].isnull().any():
        if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            median_value = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_value)
            X_test[col] = X_test[col].fillna(median_value)
        else:
            X_train[col] = X_train[col].astype(object).fillna('missing')
            X_test[col] = X_test[col].astype(object).fillna('missing')

# 统一转换为数值类型
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 划分训练/验证
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# 构建模型
inputs = tf.keras.Input(shape=(X_tr.shape[1],), name='input_features')

x = inputs
for i in range(16):
    x = tf.keras.layers.Dense(128, activation='relu', name=f'dense_{i+1}')(x)
    x = tf.keras.layers.BatchNormalization(name=f'bn_{i+1}')(x)
    x = tf.keras.layers.Dropout(0.1, name=f'dropout_{i+1}')(x)

outputs = tf.keras.layers.Dense(1, activation='linear', name='price_output')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='tf_price_model')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error',
    metrics=['mean_absolute_error', 'mean_squared_error']
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
]

history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=512,
    callbacks=callbacks,
    verbose=2
)

# 预测
preds = model.predict(X_test_scaled, batch_size=512).flatten()
preds = np.maximum(preds, 0)

# 生成提交文件
submission = pd.DataFrame({
    'SaleID': test_df['SaleID'],
    'price': preds.astype(int)
})
submission.to_csv('tensorflow_price_prediction.csv', index=False)

print('预测文件已生成: tensorflow_price_prediction.csv')
print('验证集最后一次结果:')
val_loss, val_mae, val_mse = model.evaluate(X_val, y_val, verbose=0)
print(f'val_mae={val_mae:.2f}, val_mse={val_mse:.2f}')

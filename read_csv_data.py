import pandas as pd

# 文件路径
test_file = 'used_car_testB_20200421.csv'
train_file = 'used_car_train_20200313.csv'

# 读取测试集前10行
print("=" * 60)
print("读取测试集 (used_car_testB_20200421.csv) 前10行")
print("=" * 60)
test_df = pd.read_csv(test_file, sep=' ')
print("\n测试集列字段:")
print(f"总列数: {len(test_df.columns)}")
print(f"列名: {list(test_df.columns)}")
print("\n前10行数据:")
print(test_df.head(10))
print(f"\n测试集数据形状: {test_df.shape}")

# 读取训练集前10行
print("\n" + "=" * 60)
print("读取训练集 (used_car_train_20200313.csv) 前10行")
print("=" * 60)
train_df = pd.read_csv(train_file, sep=' ')
print("\n训练集列字段:")
print(f"总列数: {len(train_df.columns)}")
print(f"列名: {list(train_df.columns)}")
print("\n前10行数据:")
print(train_df.head(10))
print(f"\n训练集数据形状: {train_df.shape}")

# 比较两个数据集的列信息
print("\n" + "=" * 60)
print("数据集对比")
print("=" * 60)
test_cols = set(test_df.columns)
train_cols = set(train_df.columns)

if test_cols == train_cols:
    print("✓ 两个数据集的列完全相同")
else:
    print("⚠ 两个数据集的列不同")
    print(f"仅在测试集中: {test_cols - train_cols}")
    print(f"仅在训练集中: {train_cols - test_cols}")

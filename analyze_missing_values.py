import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')

print("=" * 70)
print("缺失值统计分析")
print("=" * 70)

# 训练集缺失值统计
print("\n【训练集】used_car_train_20200313.csv")
print("-" * 70)
train_missing = train_df.isnull().sum()
train_missing_pct = (train_df.isnull().sum() / len(train_df)) * 100
train_missing_df = pd.DataFrame({
    '缺失数量': train_missing,
    '缺失比例(%)': train_missing_pct
})
train_missing_df = train_missing_df[train_missing_df['缺失数量'] > 0]

if len(train_missing_df) == 0:
    print("✓ 训练集无缺失值")
else:
    print("缺失值详情：")
    print(train_missing_df)

# 测试集缺失值统计
print("\n【测试集】used_car_testB_20200421.csv")
print("-" * 70)
test_missing = test_df.isnull().sum()
test_missing_pct = (test_df.isnull().sum() / len(test_df)) * 100
test_missing_df = pd.DataFrame({
    '缺失数量': test_missing,
    '缺失比例(%)': test_missing_pct
})
test_missing_df = test_missing_df[test_missing_df['缺失数量'] > 0]

if len(test_missing_df) == 0:
    print("✓ 测试集无缺失值")
else:
    print("缺失值详情：")
    print(test_missing_df)

# 检查特殊值（如'-'或'nan'字符串）
print("\n" + "=" * 70)
print("检查特殊值（字符串缺失）")
print("=" * 70)

for col in train_df.columns:
    if train_df[col].dtype == 'object':
        special_count = (train_df[col] == '-').sum()
        if special_count > 0:
            print(f"\n训练集 {col}：包含 {special_count} 个 '-' 值")
            print(f"  - 占比：{(special_count / len(train_df)) * 100:.2f}%")

for col in test_df.columns:
    if test_df[col].dtype == 'object':
        special_count = (test_df[col] == '-').sum()
        if special_count > 0:
            print(f"\n测试集 {col}：包含 {special_count} 个 '-' 值")
            print(f"  - 占比：{(special_count / len(test_df)) * 100:.2f}%")

# 生成可视化
print("\n" + "=" * 70)
print("生成可视化图表...")
print("=" * 70)

# 创建图表
fig = plt.figure(figsize=(16, 10))

# 1. 训练集缺失值热力图
ax1 = plt.subplot(2, 2, 1)
train_missing_vis = train_df.isnull().astype(int)
sns.heatmap(train_missing_vis.iloc[:100, :], cbar=True, cmap='YlOrRd', ax=ax1, 
            yticklabels=False, xticklabels=True)
ax1.set_title('训练集前100行缺失值热力图\n（红色表示缺失）', fontsize=12, fontweight='bold')
ax1.set_xlabel('字段名称')

# 2. 测试集缺失值热力图
ax2 = plt.subplot(2, 2, 2)
test_missing_vis = test_df.isnull().astype(int)
sns.heatmap(test_missing_vis.iloc[:100, :], cbar=True, cmap='YlOrRd', ax=ax2,
            yticklabels=False, xticklabels=True)
ax2.set_title('测试集前100行缺失值热力图\n（红色表示缺失）', fontsize=12, fontweight='bold')
ax2.set_xlabel('字段名称')

# 3. 训练集缺失值柱状图
ax3 = plt.subplot(2, 2, 3)
train_all_missing = train_df.isnull().sum()
train_missing_cols = train_all_missing[train_all_missing > 0]
if len(train_missing_cols) > 0:
    train_missing_cols.plot(kind='barh', ax=ax3, color='#FF6B6B')
    ax3.set_xlabel('缺失值数量')
    ax3.set_title('训练集缺失值统计', fontsize=12, fontweight='bold')
else:
    ax3.text(0.5, 0.5, '无缺失值', ha='center', va='center', fontsize=14, 
             transform=ax3.transAxes, color='green', fontweight='bold')
    ax3.set_title('训练集缺失值统计', fontsize=12, fontweight='bold')
    ax3.axis('off')

# 4. 测试集缺失值柱状图
ax4 = plt.subplot(2, 2, 4)
test_all_missing = test_df.isnull().sum()
test_missing_cols = test_all_missing[test_all_missing > 0]
if len(test_missing_cols) > 0:
    test_missing_cols.plot(kind='barh', ax=ax4, color='#4ECDC4')
    ax4.set_xlabel('缺失值数量')
    ax4.set_title('测试集缺失值统计', fontsize=12, fontweight='bold')
else:
    ax4.text(0.5, 0.5, '无缺失值', ha='center', va='center', fontsize=14,
             transform=ax4.transAxes, color='green', fontweight='bold')
    ax4.set_title('测试集缺失值统计', fontsize=12, fontweight='bold')
    ax4.axis('off')

plt.tight_layout()
plt.savefig('缺失值可视化分析.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存：缺失值可视化分析.png")

# 生成详细的缺失值统计表
print("\n" + "=" * 70)
print("生成缺失值统计表...")
print("=" * 70)

fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))

# 训练集缺失值比例
train_missing_stats = pd.DataFrame({
    '字段': train_df.columns,
    '缺失数量': train_df.isnull().sum().values,
    '缺失比例(%)': ((train_df.isnull().sum() / len(train_df)) * 100).values
})
train_missing_stats = train_missing_stats.sort_values('缺失数量', ascending=False)

ax5.axis('tight')
ax5.axis('off')
table1 = ax5.table(cellText=train_missing_stats[train_missing_stats['缺失数量'] > 0].values if (train_missing_stats['缺失数量'] > 0).any() else [['全部字段无缺失值']],
                   colLabels=['字段', '缺失数量', '缺失比例(%)'],
                   cellLoc='center',
                   loc='center',
                   bbox=[0, 0, 1, 1])
table1.auto_set_font_size(False)
table1.set_fontsize(9)
table1.scale(1, 2)
ax5.set_title('训练集缺失值统计表', fontsize=12, fontweight='bold', pad=20)

# 测试集缺失值比例
test_missing_stats = pd.DataFrame({
    '字段': test_df.columns,
    '缺失数量': test_df.isnull().sum().values,
    '缺失比例(%)': ((test_df.isnull().sum() / len(test_df)) * 100).values
})
test_missing_stats = test_missing_stats.sort_values('缺失数量', ascending=False)

ax6.axis('tight')
ax6.axis('off')
table2 = ax6.table(cellText=test_missing_stats[test_missing_stats['缺失数量'] > 0].values if (test_missing_stats['缺失数量'] > 0).any() else [['全部字段无缺失值']],
                   colLabels=['字段', '缺失数量', '缺失比例(%)'],
                   cellLoc='center',
                   loc='center',
                   bbox=[0, 0, 1, 1])
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 2)
ax6.set_title('测试集缺失值统计表', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('缺失值统计表.png', dpi=300, bbox_inches='tight')
print("✓ 表格已保存：缺失值统计表.png")

print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)

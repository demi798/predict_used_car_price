import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# 设置中文字体 - 自动检测系统可用字体
def setup_chinese_font():
    # macOS 系统中文字体
    font_names = ['SimHei', 'STHeiti', 'PingFang SC', 'Heiti TC', 
                  'Songti SC', 'STSongti', 'Kaiti SC', 'Microsoft YaHei']
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font_name in font_names:
        if any(font_name in f for f in available_fonts):
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            return font_name
    
    # 如果没有找到中文字体，使用系统默认
    plt.rcParams['font.sans-serif'] = fm.rcParams['font.sans-serif']
    return None

setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("=" * 80)
print("EDA (探索性数据分析)")
print("=" * 80)

test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')

print("\n【数据集基本信息】")
print("-" * 80)
print(f"训练集: {train_df.shape[0]:,} 行, {train_df.shape[1]} 列")
print(f"测试集: {test_df.shape[0]:,} 行, {test_df.shape[1]} 列")

# 1. 基本统计
print("\n【训练集基本统计信息】")
print("-" * 80)
print(train_df.describe().round(3).to_string())

# 2. 数据类型统计
print("\n\n【数据类型统计】")
print("-" * 80)
print(train_df.dtypes.value_counts())

# 3. 数值型特征分析
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n\n【数值型特征 ({len(numeric_cols)} 个)】")
print("-" * 80)
print(numeric_cols)

# 4. 分类特征分析
print(f"\n\n【分类特征分析】")
print("-" * 80)
for col in train_df.select_dtypes(include=['object']).columns:
    print(f"\n{col}:")
    print(f"  - 唯一值数: {train_df[col].nunique()}")
    print(f"  - 前5个值: {train_df[col].value_counts().head().to_dict()}")

# 5. 目标变量（price）分析
print(f"\n\n【目标变量 - price 分析】")
print("-" * 80)
print(f"最小值: {train_df['price'].min()}")
print(f"最大值: {train_df['price'].max()}")
print(f"平均值: {train_df['price'].mean():.2f}")
print(f"中位数: {train_df['price'].median():.2f}")
print(f"标准差: {train_df['price'].std():.2f}")
print(f"偏度: {train_df['price'].skew():.3f}")
print(f"峰度: {train_df['price'].kurtosis():.3f}")

# 6. 相关性分析
print(f"\n\n【与price相关性最强的特征】")
print("-" * 80)
correlation = train_df[numeric_cols].corr()['price'].sort_values(ascending=False)
print(correlation.head(10).to_string())

# ============ 生成可视化 ============
print(f"\n\n正在生成可视化图表...")
print("=" * 80)

# 创建大图表
fig = plt.figure(figsize=(20, 24))

# 1. 价格分布
ax1 = plt.subplot(5, 3, 1)
train_df['price'].hist(bins=100, ax=ax1, color='#FF6B6B', edgecolor='black')
ax1.set_title('价格分布（直方图）', fontsize=12, fontweight='bold')
ax1.set_xlabel('价格')
ax1.set_ylabel('频数')

# 2. 价格分布（KDE）
ax2 = plt.subplot(5, 3, 2)
train_df['price'].plot(kind='density', ax=ax2, color='#4ECDC4', linewidth=2)
ax2.set_title('价格分布（密度图）', fontsize=12, fontweight='bold')
ax2.set_xlabel('价格')

# 3. 价格箱线图
ax3 = plt.subplot(5, 3, 3)
train_df.boxplot(column='price', ax=ax3)
ax3.set_title('价格箱线图（离群值检测）', fontsize=12, fontweight='bold')

# 4. 里程数分布
ax4 = plt.subplot(5, 3, 4)
train_df['kilometer'].hist(bins=100, ax=ax4, color='#95E1D3', edgecolor='black')
ax4.set_title('行驶里程分布', fontsize=12, fontweight='bold')
ax4.set_xlabel('里程数')
ax4.set_ylabel('频数')

# 5. 功率分布
ax5 = plt.subplot(5, 3, 5)
train_df['power'].hist(bins=100, ax=ax5, color='#F38181', edgecolor='black')
ax5.set_title('发动机功率分布', fontsize=12, fontweight='bold')
ax5.set_xlabel('功率')
ax5.set_ylabel('频数')

# 6. 注册年份分布
ax6 = plt.subplot(5, 3, 6)
train_df['regDate'] = train_df['regDate'].astype(str)
train_df['year'] = train_df['regDate'].str[:4].astype(int)
train_df['year'].hist(bins=50, ax=ax6, color='#FFD93D', edgecolor='black')
ax6.set_title('车辆注册年份分布', fontsize=12, fontweight='bold')
ax6.set_xlabel('年份')
ax6.set_ylabel('频数')

# 7. 品牌Top 15
ax7 = plt.subplot(5, 3, 7)
train_df['brand'].value_counts().head(15).plot(kind='barh', ax=ax7, color='#6BCB77')
ax7.set_title('品牌分布（Top 15）', fontsize=12, fontweight='bold')
ax7.set_xlabel('数量')

# 8. 车体类型
ax8 = plt.subplot(5, 3, 8)
train_df['bodyType'].value_counts().plot(kind='bar', ax=ax8, color='#4D96FF')
ax8.set_title('车体类型分布', fontsize=12, fontweight='bold')
ax8.set_ylabel('数量')
ax8.tick_params(axis='x', rotation=45)

# 9. 燃料类型
ax9 = plt.subplot(5, 3, 9)
train_df['fuelType'].value_counts().plot(kind='bar', ax=ax9, color='#FF6348')
ax9.set_title('燃料类型分布', fontsize=12, fontweight='bold')
ax9.set_ylabel('数量')
ax9.tick_params(axis='x', rotation=45)

# 10. 卖家类型
ax10 = plt.subplot(5, 3, 10)
seller_type = {0: '个人', 1: '经销商'}
seller_data = train_df['seller'].map(seller_type).value_counts()
ax10.pie(seller_data, labels=seller_data.index, autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
ax10.set_title('卖家类型分布', fontsize=12, fontweight='bold')

# 11. 变速箱类型
ax11 = plt.subplot(5, 3, 11)
gearbox_type = {0.0: '手动', 1.0: '自动'}
gearbox_data = train_df['gearbox'].map(gearbox_type).value_counts()
ax11.pie(gearbox_data, labels=gearbox_data.index, autopct='%1.1f%%', colors=['#FFD93D', '#6BCB77'])
ax11.set_title('变速箱类型分布', fontsize=12, fontweight='bold')

# 12. 是否有损伤
ax12 = plt.subplot(5, 3, 12)
damage_type = {0.0: '无损伤', 1.0: '有损伤'}
damage_data = train_df['notRepairedDamage'].map(damage_type).value_counts()
ax12.pie(damage_data, labels=damage_data.index, autopct='%1.1f%%', colors=['#95E1D3', '#F38181'])
ax12.set_title('未修复损伤分布', fontsize=12, fontweight='bold')

# 13. 价格vs里程数
ax13 = plt.subplot(5, 3, 13)
sample_idx = np.random.choice(len(train_df), min(5000, len(train_df)), replace=False)
ax13.scatter(train_df.iloc[sample_idx]['kilometer'], 
             train_df.iloc[sample_idx]['price'], 
             alpha=0.3, s=10, color='#4ECDC4')
ax13.set_title('价格 vs 里程数', fontsize=12, fontweight='bold')
ax13.set_xlabel('里程数')
ax13.set_ylabel('价格')

# 14. 价格vs功率
ax14 = plt.subplot(5, 3, 14)
ax14.scatter(train_df.iloc[sample_idx]['power'], 
             train_df.iloc[sample_idx]['price'], 
             alpha=0.3, s=10, color='#FF6B6B')
ax14.set_title('价格 vs 功率', fontsize=12, fontweight='bold')
ax14.set_xlabel('功率')
ax14.set_ylabel('价格')

# 15. 价格vs车龄
ax15 = plt.subplot(5, 3, 15)
train_df['car_age'] = 2020 - train_df['year']
ax15.scatter(train_df.iloc[sample_idx]['car_age'], 
             train_df.iloc[sample_idx]['price'], 
             alpha=0.3, s=10, color='#6BCB77')
ax15.set_title('价格 vs 车龄', fontsize=12, fontweight='bold')
ax15.set_xlabel('车龄（年）')
ax15.set_ylabel('价格')

plt.tight_layout()
plt.savefig('EDA_综合分析.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: EDA_综合分析.png")

# ============ 相关性热力图 ============
fig2, ax = plt.subplots(figsize=(14, 12))

# 计算相关系数矩阵
numeric_data = train_df[numeric_cols].copy()
corr_matrix = numeric_data.corr()

# 绘制热力图
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, ax=ax, cbar_kws={'label': '相关系数'}, 
            vmin=-1, vmax=1, annot_kws={'size': 8})
ax.set_title('数值特征相关性矩阵热力图', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('相关性热力图.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: 相关性热力图.png")

# ============ 特征分布直方图 ============
numeric_features = numeric_cols[:12]  # 选择前12个特征
fig3 = plt.figure(figsize=(18, 12))

for idx, col in enumerate(numeric_features, 1):
    ax = plt.subplot(3, 4, idx)
    train_df[col].hist(bins=50, ax=ax, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}分布', fontsize=10, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('频数')

plt.tight_layout()
plt.savefig('特征分布直方图.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: 特征分布直方图.png")

# ============ 统计摘要 ============
print("\n" + "=" * 80)
print("【EDA 分析摘要】")
print("=" * 80)
print(f"\n✓ 数据集大小: 训练集 {train_df.shape[0]:,} 行, 测试集 {test_df.shape[0]:,} 行")
print(f"✓ 特征数量: {train_df.shape[1]} 个（包含1个目标变量price）")
print(f"✓ 数值特征: {len(numeric_cols)} 个")
print(f"✓ 缺失值字段: bodyType, fuelType, gearbox, model（已在前面分析）")
print(f"\n✓ 目标变量(price)统计:")
print(f"   - 范围: {train_df['price'].min()} ~ {train_df['price'].max()}")
print(f"   - 平均: {train_df['price'].mean():.2f}")
print(f"   - 中位数: {train_df['price'].median():.2f}")
print(f"   - 标准差: {train_df['price'].std():.2f}")

print(f"\n✓ 与price相关性最强的5个特征:")
top_corr = correlation.head(6)[1:]  # 排除price本身
for i, (feat, corr_val) in enumerate(top_corr.items(), 1):
    print(f"   {i}. {feat}: {corr_val:.4f}")

print(f"\n✓ 生成的可视化文件:")
print(f"   1. EDA_综合分析.png - 15个子图综合分析")
print(f"   2. 相关性热力图.png - 特征间相关性")
print(f"   3. 特征分布直方图.png - 各特征分布")

print("\n" + "=" * 80)
print("EDA 分析完成！")
print("=" * 80)

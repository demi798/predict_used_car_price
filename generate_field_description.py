import pandas as pd

# 读取数据
test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')

# 字段描述字典
field_descriptions = {
    'SaleID': '销售ID，唯一标识符',
    'name': '汽车名称/型号编码',
    'regDate': '汽车注册日期，格式为YYYYMMDD',
    'model': '汽车型号代码',
    'brand': '汽车品牌代码',
    'bodyType': '汽车车体类型代码（如轿车、SUV等）',
    'fuelType': '燃料类型代码（如汽油、柴油等）',
    'gearbox': '变速箱类型代码（如自动、手动等）',
    'power': '发动机功率',
    'kilometer': '汽车行驶公里数',
    'notRepairedDamage': '是否有未修复的损伤',
    'regionCode': '地区代码',
    'seller': '卖家类型',
    'offerType': '报价类型',
    'creatDate': '数据创建日期',
    'price': '二手车价格（仅在训练集中）',
    'v_0 到 v_14': '经过处理的匿名特征（可能是PCA或其他特征工程得到）'
}

# 生成markdown文件
md_content = """# 二手车价格预测数据集字段说明

## 数据集概览

- **训练集**：used_car_train_20200313.csv (150,000 行，31 列)
- **测试集**：used_car_testB_20200421.csv (50,000 行，30 列)
- **数据分隔符**：空格

---

## 字段详细说明

### 基本信息字段

| 字段名称 | 数据类型 | 描述 |
|---------|--------|------|
| SaleID | int | 销售ID，唯一标识符 |
| name | int | 汽车名称/型号编码 |
| regDate | int | 汽车注册日期（格式：YYYYMMDD），如20040402表示2004年4月2日 |
| creatDate | int | 数据创建日期（格式：YYYYMMDD） |

### 汽车属性字段

| 字段名称 | 数据类型 | 描述 |
|---------|--------|------|
| brand | int | 汽车品牌代码 |
| model | float | 汽车型号代码 |
| bodyType | float | 汽车车体类型代码（0/1/2/3/4等编码）|
| fuelType | float | 燃料类型代码（0/1/2等编码，如汽油、柴油、混合动力等）|
| gearbox | float | 变速箱类型代码（0/1等编码，0表示手动，1表示自动）|
| power | int | 发动机功率（单位：马力或kW） |
| kilometer | float | 汽车行驶公里数 |

### 交易信息字段

| 字段名称 | 数据类型 | 描述 |
|---------|--------|------|
| seller | int | 卖家类型（0或1，区分个人卖家或经销商） |
| offerType | int | 报价类型（0或1） |
| notRepairedDamage | int/str | 是否有未修复的损伤（0/1或-表示缺失值） |
| regionCode | int | 地区代码（代表不同的销售地区） |

### 目标变量

| 字段名称 | 数据类型 | 描述 | 备注 |
|---------|--------|------|------|
| price | float | 二手车价格 | 仅在训练集中出现，是预测目标 |

### 工程特征字段

| 字段名称 | 数据类型 | 描述 |
|---------|--------|------|
| v_0 - v_14 | float | 经过处理的匿名特征（共15个）|
| | | 可能来自PCA（主成分分析）或其他特征工程处理 |
| | | 数值已标准化，取值范围约为 [-5, 3] |

---

## 数据统计信息

### 训练集统计
"""

# 添加训练集统计信息
md_content += "\n**数据形状**：" + str(train_df.shape) + "\n\n"
md_content += "**数据类型**：\n```\n"
md_content += str(train_df.dtypes)
md_content += "\n```\n\n"

md_content += "**缺失值统计**：\n```\n"
md_content += str(train_df.isnull().sum())
md_content += "\n```\n\n"

md_content += "**基本统计量**：\n```\n"
md_content += str(train_df.describe().round(2))
md_content += "\n```\n\n"

### 测试集统计
md_content += "### 测试集统计\n\n"
md_content += "**数据形状**：" + str(test_df.shape) + "\n\n"
md_content += "**数据类型**：\n```\n"
md_content += str(test_df.dtypes)
md_content += "\n```\n\n"

md_content += "**缺失值统计**：\n```\n"
md_content += str(test_df.isnull().sum())
md_content += "\n```\n\n"

md_content += "**基本统计量**：\n```\n"
md_content += str(test_df.describe().round(2))
md_content += "\n```\n\n"

# 添加样本数据
md_content += """
---

## 样本数据

### 训练集前5行
"""
md_content += "\n```\n"
md_content += str(train_df.head())
md_content += "\n```\n\n"

md_content += """### 测试集前5行
"""
md_content += "\n```\n"
md_content += str(test_df.head())
md_content += "\n```\n\n"

md_content += """
---

## 关键特征说明

### 时间特征
- **regDate**：注册日期，反映车龄信息
- **creatDate**：数据创建日期

### 性能特征
- **power**：发动机功率，直接影响车价
- **kilometer**：行驶里程，与车况密切相关

### 品牌与型号
- **brand**、**model**：品牌和型号信息
- 不同品牌与型号的价值差异较大

### 匿名特征
- **v_0 到 v_14**：这15个特征是经过数据预处理的，原始含义已隐藏
- 可能包含品牌、排量、车型等经过编码或降维处理的信息

---

## 数据预处理建议

1. **日期字段转换**：将 regDate 和 creatDate 转换为年、月、日，或计算车龄
2. **缺失值处理**：检查 notRepairedDamage 等字段的缺失值
3. **分类变量编码**：验证 brand、bodyType、fuelType 等是否需要进一步处理
4. **特征选择**：结合 v_0-v_14 的工程特征与原始特征建模

---

*生成时间：2026年4月20日*
"""

# 写入markdown文件
with open('字段含义说明.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print("✓ Markdown文件已生成：字段含义说明.md")
print(f"✓ 文件大小：{len(md_content)} 字符")

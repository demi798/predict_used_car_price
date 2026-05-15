# 二手车价格预测项目 - 文件说明文档

## 📋 项目概述

这是一个天池竞赛项目([竞赛链接](https://tianchi.aliyun.com/competition/entrance/231784))，整个项目基于机器学习的二手车价格预测项目，使用了XGBoost和CatBoost等梯度提升模型。通过综合的特征工程、数据清洗和模型优化，实现了高精度的价格预测。

**最终验证集MAE: 521.70元**

---

## 📁 文件结构说明

### 📊 数据文件


| 文件名                        | 说明                              | 大小类型 |
| ----------------------------- | --------------------------------- | -------- |
| `used_car_train_20200313.csv` | 训练集数据（150,000条记录，31列） | 原始数据 |
| `used_car_testB_20200421.csv` | 测试集数据（50,000条记录，30列）  | 原始数据 |
| `used_car_sample_submit.csv`  | 样本提交文件格式参考              | 参考模板 |

### 🔧 核心Python脚本

#### 数据分析与预处理


| 文件名                          | 功能                | 关键输出                 |
| ------------------------------- | ------------------- | ------------------------ |
| `read_csv_data.py`              | 数据读取和初步探索  | 数据形状、类型、统计信息 |
| `eda_analysis.py`               | 探索性数据分析(EDA) | 生成EDA可视化图表        |
| `analyze_missing_values.py`     | 缺失值分析与处理    | 缺失值统计和填充策略     |
| `generate_field_description.py` | 生成字段含义说明    | 字段描述文档             |

#### 特征工程


| 文件名                                   | 功能                                                     | 特征数量变化                                                       |
| ---------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------ |
| `feature_engineering_and_catboost.py`    | 初始特征工程与CatBoost模型                               | 31 → 45个特征                                                     |
| `xgboost_advanced_features.py`           | **最优方案**：高级特征工程 + 异常值清洗 + 交叉检测       | 31 → 72个特征，MAE: 520.6327（最优）                             |
| `catboost_advanced_features.py`          | CatBoost高级特征版本                                     | 31 → 72个特征，MAE: 468.43<br />（实际MAE：1170，存在严重过拟合） |
| `tensorflow_price_prediction_refined.py` | TensorFlow深度学习回归模型，沿用CatBoost/XGBoost特征工程 | 59个特征，验证集MAE: 521.6963524.80                                |

#### 模型与预测


| 文件名                                   | 模型类型                    | 验证集MAE |
| ---------------------------------------- | --------------------------- | --------- |
| `xgboost_predict.py`                     | 基础XGBoost                 | -         |
| `decision_tree_predict.py`               | 决策树对比                  | -         |
| `xgboost_optimized.py`                   | XGBoost优化版               | -         |
| `xgboost_optimized_ensemble.py`          | XGBoost集成版（多模型融合） | -         |
| `tensorflow_price_prediction_refined.py` | TensorFlow深度学习回归模型  | 524.80    |

### 📈 结果与输出文件

#### 预测结果


| 文件名                                    | 包含内容             | 来源                                   |
| ----------------------------------------- | -------------------- | -------------------------------------- |
| `xgboost_prediction_result.csv`           | 测试集预测结果       | xgboost_predict.py                     |
| `xgboost_advanced_features_result.csv`    | 高级特征XGBoost预测  | xgboost_advanced_features.py           |
| `xgboost_optimized_result.csv`            | 优化版XGBoost预测    | xgboost_optimized.py                   |
| `xgboost_optimized_ensemble_result.csv`   | 集成模型预测         | xgboost_optimized_ensemble.py          |
| `catboost_advanced_features_result.csv`   | CatBoost高级特征预测 | catboost_advanced_features.py          |
| `tensorflow_price_prediction_refined.csv` | TensorFlow预测结果   | tensorflow_price_prediction_refined.py |
| `prediction_result.csv`                   | 最终预测结果         | 综合最优模型                           |

#### 训练曲线与可视化


| 文件名                                          | 说明                   | 指标             |
| ----------------------------------------------- | ---------------------- | ---------------- |
| `xgboost_training_curve.png`                    | 基础XGBoost训练曲线    | 训练/验证RMSE    |
| `xgboost_optimized_training_curve.png`          | 优化版XGBoost训练曲线  | 训练/验证RMSE    |
| `catboost_advanced_features_training_curve.png` | CatBoost训练曲线       | 训练/验证RMSE    |
| `EDA_综合分析.png`                              | 数据分布综合分析       | 特征分布与相关性 |
| `特征分布直方图.png`                            | 特征值分布直方图       | 数值特征分布     |
| `相关性热力图.png`                              | 特征与目标相关性热力图 | Pearson相关系数  |
| `缺失值可视化分析.png`                          | 缺失值分布可视化       | 缺失值比例       |
| `缺失值统计表.png`                              | 缺失值统计表           | 各字段缺失数     |

### 📝 文档文件


| 文件名                | 内容                 | 关键信息           |
| --------------------- | -------------------- | ------------------ |
| `字段含义说明.md`     | 数据集字段的详细说明 | 31个原始字段定义   |
| `特征工程.md`         | 特征工程方案详细说明 | 特征创建方法与原理 |
| `优化方案总结报告.md` | 模型优化方案总结     | 优化策略与结果对比 |
| `快速参考卡.md`       | 项目快速查找手册     | 关键参数与命令     |
| `项目完成总结.py`     | 项目总结脚本         | 最终统计与分析     |

### 🗂️ 其他文件


| 文件名           | 说明                  |
| ---------------- | --------------------- |
| `catboost_info/` | CatBoost训练日志目录  |
| `__pycache__/`   | Python缓存文件        |
| `.venv/`         | 虚拟环境目录1         |
| `.venv-1/`       | 虚拟环境目录2         |
| `.DS_Store`      | Mac系统文件（可忽略） |

---

## 🎯 核心特征工程方案（最优方案）

### 特征类型与数量

```
原始特征: 31个
↓
新增特征：
├─ 时间特征（5个）：car_age_segment, km_per_year, reg_season, time_diff, reg_year_period
├─ 品牌特征（7个）：brand_avg_price, brand_price_std, brand_count, brand_tier, 
│               brand_model_avg_price, brand_model_count, region_brand_avg_price
├─ 车况特征（8个）：km_segment, power_segment, power_km_ratio, power_to_brand_price,
│               power_per_age, km_per_year_segment, price_per_km, log_*
├─ 异常标志（6个）：notRepaired_is_missing, km_per_year_high, new_car_high_km,
│               power_is_zero, age_region_anomaly, age_model_anomaly
└─ 交互与编码（8个）：power_age, km_age, log_power, log_km, *_encoded

总计: 72个特征
```

### 数据清洁与处理

1. **异常值清洗**：使用IQR方法移除极端值（kilometer, power, regDate）

   - 训练集：150,000 → 120,720条（移除2.2%异常样本）
2. **缺失值处理**

   - 分类特征：填充为'missing'
   - 数值特征：填充为中位数
3. **特征编码**

   - 目标编码：bodyType, fuelType, gearbox, regionCode
   - 顺序编码：品牌分级、车龄分段等
4. **交叉异常检测**

   - car_age × regionCode：检测地区-车龄的异常组合
   - car_age × model：检测型号-车龄的异常价格

---

## 🤖 模型性能对比


| 模型              | 验证集MAE  | 验证集RMSE  | 验证集R²  | 迭代轮数  |
| ----------------- | ---------- | ----------- | ---------- | --------- |
| 基础XGBoost       | -          | -           | -          | 2000      |
| XGBoost优化版     | -          | -           | -          | 2000      |
| XGBoost高级特征   | 643.04     | 1334.94     | 0.9675     | 2000      |
| **XGBoost最优版** | **522.17** | **1198.00** | **0.9738** | **10436** |
| CatBoost高级特征  | 468.43     | 970.23      | 0.9588     | 1999      |

**✅ 最优模型：** `xgboost_advanced_features.py`

- 验证集MAE: **522.17元**
- 相比基础版本改善约27%

---

## 🔑 特征重要性Top 10


| 排名 | 特征名         | 重要度 | 说明                         |
| ---- | -------------- | ------ | ---------------------------- |
| 1    | v_3            | 24.67% | 匿名特征3（原始数据）        |
| 2    | v_0            | 21.22% | 匿名特征0（原始数据）        |
| 3    | v_12           | 15.72% | 匿名特征12（原始数据）       |
| 4    | v_6            | 3.58%  | 匿名特征6                    |
| 5    | power_km_ratio | 2.73%  | **工程特征**：功率/里程比    |
| 6    | v_10           | 2.62%  | 匿名特征10                   |
| 7    | v_9            | 2.62%  | 匿名特征9                    |
| 8    | v_14           | 2.00%  | 匿名特征14                   |
| 9    | power          | 1.47%  | 原始功率                     |
| 10   | time_diff      | 1.61%  | **工程特征**：注册到上线延迟 |

---

## 📊 快速开始

### 1. 查看数据

```bash
python3 read_csv_data.py
```

### 2. EDA分析

```bash
python3 eda_analysis.py
```

### 3. 运行最优模型

```bash
python3 xgboost_advanced_features.py
```

### 4. 查看结果

- 预测结果：`xgboost_advanced_features_result.csv`
- 训练曲线：`catboost_advanced_features_training_curve.png`

---

## 📚 主要改进方向

### Phase 1: 基础模型

- ✅ 数据读取与清洗
- ✅ 基础特征工程（时间、品牌、车况）
- ✅ XGBoost训练

### Phase 2: 高级特征工程

- ✅ 异常值清洗（IQR方法）
- ✅ 新增统计特征（品牌-型号、地区-品牌）
- ✅ 交叉异常检测
- ✅ 细分区间特征（km_per_year_segment等）

### Phase 3: 模型优化

- ✅ 超参数调优
- ✅ 集成方法（多模型融合）
- ✅ CatBoost原生类别特征支持

---

## 🎯 关键指标

### 数据规模

- 训练集：120,720条记录（清洗后）
- 验证集：24,144条记录
- 测试集：50,000条记录
- 特征数：72个

### 模型性能

- **最优验证集MAE：522.17元**
- 模型收敛轮数：10,436轮
- 早停等待轮数：20轮

### 特征工程收益

- 基础特征 → 高级特征：31 → 72 (+132%)
- 验证集MAE改善：643.04 → 521.17 (-18.8%)

---

## 📖 使用说明

### 推荐阅读顺序

1. `快速参考卡.md` - 快速了解项目
2. `字段含义说明.md` - 理解数据含义
3. `特征工程.md` - 深入特征设计
4. `优化方案总结报告.md` - 了解优化过程

### 推荐执行顺序

1. `read_csv_data.py` - 数据探索
2. `eda_analysis.py` - 可视化分析
3. `xgboost_advanced_features.py` - 运行最优模型

---

## ⚙️ 环境要求

```
Python >= 3.8
catboost >= 1.0
xgboost >= 1.5
pandas >= 1.3
numpy >= 1.21
scikit-learn >= 0.24
matplotlib >= 3.4
```

---

## 📌 更新时间

- **最后更新**：2026年4月29日
- **最优模型运行时间**：约15分钟
- **验证集MAE**：522.17元

---

## 📞 项目统计

- **总文件数**：35+个
- **Python脚本**：11个
- **数据文件**：3个
- **结果文件**：6个
- **可视化**：8个
- **文档**：5个

---

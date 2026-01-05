# 医学影像组学与剂量组学预测放射性口干

基于多模态医学影像（CT影像和剂量分布）的机器学习项目，用于预测鼻咽癌患者放疗后口干症（Xerostomia）。

## 项目概述

本项目旨在通过提取和分析腮腺及颌下腺的影像组学（Radiomics）和剂量组学（Dosiomics）特征，结合机器学习方法，预测患者放疗后口干症的发生风险。

### 研究背景

- **疾病**：鼻咽癌（NPC）患者放疗后口干症
- **目标器官**：腮腺（Parotid Glands）、颌下腺（Submandibular Glands）
- **样本量**：383例患者
- **任务类型**：二分类（发生口干 vs 未发生口干）

### 技术特点

- **IBSI标准**：遵循IBSI（Image Biomarker Standardisation Initiative）标准进行特征提取
- **多模态融合**：CT影像组学 + 剂量组学
- **稳定特征选择**：基于交叉验证的LASSO稳定特征选择
- **严格验证**：训练/验证/测试三分离，避免数据泄漏

## 技术栈

| 类别                   | 技术                               |
| ---------------------- | ---------------------------------- |
| **编程语言**     | Python 3.11/3.13                   |
| **医学影像处理** | SimpleITK, nibabel, pydicom        |
| **特征提取**     | PyRadiomics (IBSI-compliant)       |
| **机器学习**     | scikit-learn, XGBoost              |
| **数据处理**     | numpy, pandas, scipy               |
| **模型评估**     | bootstrap CI, calibration analysis |

## 项目结构

```
x/
├── README.md                    # 项目说明文档
├── UniqueSamples.xlsx           # 患者标签数据
│
├── pipeline.py                  # 特征提取pipeline
├── lasso_stable_score.py        # 稳定特征选择
├── make_splits.py               # 数据集划分
├── metrics_utils.py             # 评估指标工具
│
├── LR/                          # 逻辑回归模型
│   ├── lr_train.py             # 训练脚本
│   └── omics/                  # 模型输出
│
├── SVM/                         # 支持向量机模型
│   ├── svm_train.py            # 训练脚本
│   └── omics/                  # 模型输出
│
├── XGB/                         # XGBoost模型
│   ├── xgb_train.py            # 训练脚本
│   └── omics/                  # 模型输出
│
├── 383nii/                      # 原始CT和Dose数据
├── nii/                         # 器官ROI masks
├── CT_crop/                     # 裁剪后的CT图像
├── Dose_crop/                   # 裁剪后的剂量图
│
└── outputs/                     # 输出目录
    ├── curation/                # 数据清洗和标签
    ├── features/                # 提取的特征
    ├── selection/               # 特征筛选结果
    └── splits/                  # 数据集划分
```

## 环境配置

### 1. 创建虚拟环境

```bash
# 使用Python 3.11
python3.11 -m venv .venv311
source .venv311/bin/activate

# 或使用Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：

```
SimpleITK>=2.0.0
pydicom>=2.0.0
radiomics>=3.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
scipy>=1.7.0
tqdm>=4.62.0
openpyxl>=3.0.0
joblib>=1.1.0
```

## 数据准备

### 输入数据格式

```
383nii/
└── HXXXXXX/                    # 患者目录
    ├── CT.nii.gz               # CT图像（NIfTI格式）
    ├── RD.nii.gz               # 剂量分布（NIfTI格式，单位：Gy）
    └── plan.json               # 治疗计划文件

nii/
└── HXXXXXX/                    # 患者目录
    ├── parotid_L.nii.gz        # 左腮腺mask
    ├── parotid_R.nii.gz        # 右腮腺mask
    ├── submand_L.nii.gz        # 左颌下腺mask
    └── submand_R.nii.gz        # 右颌下腺mask

UniqueSamples.xlsx              # 患者标签
├── case_id                     # 病例ID
└── y                           # 标签（0=阴性，1=阳性）
```

### 数据要求

- **CT图像**：

  - 格式：NIfTI (.nii.gz)
  - 单位：Hounsfield Unit (HU)
  - 范围：-1000 ~ 1000 HU
- **剂量图**：

  - 格式：NIfTI (.nii.gz)
  - 单位：Gray (Gy)
  - 需要DoseGridScaling校正
- **ROI mask**：

  - 格式：NIfTI (.nii.gz)
  - 值：0（背景）或 1（前景）
  - 需要提取最大连通域

## 快速开始

### 1. 数据集划分

```bash
# 生成三划分（70%训练，20%验证，10%测试）
python make_splits.py --train-size 0.7 --val-size 0.2 --test-size 0.1
```

输出：

```
outputs/splits/
├── threeway_train_ids.csv      # 训练集 (266样本)
├── threeway_val_ids.csv        # 验证集 (58样本)
└── threeway_test_ids.csv       # 测试集 (57样本)
```

### 2. 特征提取

```bash
# 提取所有382例的radiomics和dosiomics特征
python pipeline.py

# 仅处理前10例（测试用）
python pipeline.py --limit 10

# 仅准备ML-ready表格
python pipeline.py --prepare-only

# 增加对称特征（左右腺体统计）
python pipeline.py --augment-sym
```

输出：

```
outputs/features/
├── ct_radiomics_parotid.csv           # 腮腺CT特征
├── ct_radiomics_submand.csv           # 颌下腺CT特征
├── dosiomics_parotid.csv              # 腮腺剂量特征
├── dosiomics_submand.csv              # 颌下腺剂量特征
├── dvh.csv                            # DVH指标
├── ml_meta.json                       # 提取参数记录
└── roi_qc.csv                         # ROI质控报告
```

### 3. 特征选择

```bash
# CT radiomics特征选择（5折交叉验证，至少3/5折叠）
python lasso_stable_score.py \
  --modality ct \
  --pipeline ibsi_utest_lasso \
  --folds 5 \
  --seed 42

# Dose dosiomics特征选择
python lasso_stable_score.py \
  --modality dose \
  --pipeline ibsi_utest_lasso \
  --folds 5 \
  --seed 42

# 在训练集+验证集上筛选（避免测试集泄漏）
python lasso_stable_score.py \
  --modality ct \
  --restrict-ids-path outputs/splits/threeway_trainval_ids.csv
```

输出：

```
outputs/selection/
├── ct_ibsi_utest_lasso_seed42_k5_trainonly/
│   ├── stable_features_min3of5.csv          # 稳定特征列表
│   ├── stable_features_frequency.csv        # 特征频率
│   ├── run_meta.json                        # 运行元数据
│   └── fold1/ ... fold5/                    # 各折详情
│
└── dose_ibsi_utest_lasso_seed42_k5_trainonly/
    ├── stable_features_min3of5.csv          # 稳定特征列表
    └── ...
```

### 4. 模型训练

#### 逻辑回归

```bash
# 单模态 - CT radiomics
python LR/lr_train.py --task ct_min3of5

# 单模态 - Dose dosiomics
python LR/lr_train.py --task dose_min3of5

# 早融合 - CT + Dose
python LR/lr_train.py --task ct_dose_early_min3of5

# 自定义参数
python LR/lr_train.py \
  --task ct_min3of5 \
  --trials 50 \
  --splits 5 \
  --calibration-method isotonic
```

#### 支持向量机

```bash
python SVM/svm_train.py --task ct_min3of5
python SVM/svm_train.py --task dose_min3of5
python SVM/svm_train.py --task ct_dose_early_min3of5
```

#### XGBoost

```bash
python XGB/xgb_train.py --task ct_min3of5
python XGB/xgb_train.py --task dose_min3of5
python XGB/xgb_train.py --task ct_dose_early_min3of5
```

### 5. 模型评估

训练完成后，模型结果保存在 `LR/omics/{task}/metric_auc/`：

```
LR/omics/ct_min3of5_trainonly/metric_auc/
├── best_params.json                    # 最佳超参数
├── cv_metrics.csv                      # 交叉验证指标
├── train_oof_predictions.csv           # 训练集OOF预测
├── val_predictions.csv                 # 验证集预测
├── test_predictions.csv                # 测试集预测
├── test_metrics.json                   # 测试集指标（含95% CI）
└── lr_bundle.joblib                    # 完整模型pipeline
```

## 特征命名规范

所有特征遵循以下命名规则：

```
{organ}__{modality}__{feature_name}
```

示例：

- `parotid_L__ct__original_firstorder_Mean` - 左腮腺 CT原始一阶均值
- `parotid__sym__mean__ct__original_firstorder_Mean` - 腮腺（左右平均）CT均值
- `submand__sym__absdiff__dose__wavelet-HHH_glrlm_ShortRunHighGrayLevelEmphasis` - 颌下腺对称剂量特征

对称特征类型：

- `mean`: 左右平均
- `max`: 左右最大值
- `min`: 左右最小值
- `absdiff`: 左右绝对差值
- `reldiff`: 左右相对差值
- `absratio`: 左右绝对比值

## 评估指标

模型评估包含以下指标：

| 指标                  | 说明                     |
| --------------------- | ------------------------ |
| **AUC**         | ROC曲线下面积            |
| **AP**          | 平均精度                 |
| **Accuracy**    | 准确率                   |
| **Sensitivity** | 敏感性（召回率）         |
| **Specificity** | 特异性                   |
| **Precision**   | 精确率                   |
| **F1-Score**    | F1分数                   |
| **ECE**         | 期望校准误差             |
| **Brier Score** | Brier分数                |
| **95% CI**      | Bootstrap 2000次置信区间 |

## 最佳实践

### 1. 避免数据泄漏

**错误做法**：

```bash
# 在全部数据上筛选特征（包括测试集）
python lasso_stable_score.py --modality ct
```

**正确做法**：

```bash
# 仅在训练集+验证集上筛选特征
python lasso_stable_score.py \
  --modality ct \
  --restrict-ids-path outputs/splits/threeway_trainval_ids.csv
```

### 2. 模型选择流程

```
1. 数据划分 → make_splits.py
2. 特征提取 → pipeline.py
3. 特征选择 → lasso_stable_score.py (train+val only)
4. 模型训练 → lr_train.py/svm_train.py/xgb_train.py
5. 模型评估 → 查看test_metrics.json
```

### 3. 超参数调优

使用嵌套交叉验证：

- 外层5折：评估模型泛化性能
- 内层3折：超参数搜索

```python
# 内层搜索范围
C_values = np.logspace(-4, 4, 50)  # LR/SVM
n_estimators = [100, 200, 300]     # XGB
max_depth = [3, 5, 7]              # XGB
```

## 常见问题

### Q1: 特征提取失败怎么办？

检查以下几点：

1. NIfTI文件路径是否正确
2. ROI mask是否为空（体素数=0）
3. CT和Dose图像空间是否对齐
4. 查看outputs/features/roi_qc.csv获取详细错误信息

### Q2: 剂量值不对怎么办？

剂量可能需要校正：

```python
# 如果max(Dose) > 200，说明单位是cGy，需要除以100
if dose_max > 200:
    dose_array = dose_array / 100.0
```

### Q3: 内存不足怎么办？

限制处理样本数：

```bash
python pipeline.py --limit 50  # 只处理50例
```

### Q4: 测试集性能很差？

可能原因：

1. 训练集和测试集分布不匹配
2. 特征选择时包含了测试集（数据泄漏）
3. 过拟合

解决方案：

- 检查数据分布是否一致
- 重新做特征选择（排除测试集）
- 使用更保守的模型（减少复杂度）

## 项目当前状态

### 数据

- ✅ 383例患者数据
- ✅ CT图像和剂量图裁剪对齐
- ✅ ROI masks质量验证
- ✅ 三划分数据集（266/58/57）

### 特征

- ✅ CT radiomics特征（1288维/样本）
- ✅ Dose dosiomics特征（1288维/样本）
- ✅ DVH指标
- ✅ 对称特征增强

### 特征选择

- ✅ CT稳定特征：35个
- ✅ Dose稳定特征：31个

### 模型

- ✅ Logistic Regression
- ✅ SVM
- ✅ XGBoost

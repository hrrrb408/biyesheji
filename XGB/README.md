# XGB 放射组学建模模块

本目录包含用于放射组学二分类的 XGBoost (XGB) 建模流水线，覆盖 **CT**、**Dose** 以及 **多模态早期融合（Early Fusion，特征拼接）**。

## 1. 输入与前置文件（默认路径）
- 标签表：`outputs/curation/index_with_labels.csv`（列：`case_id`、`y`）
- 特征表：`outputs/features/*.csv`（列：`case_id` + 特征列）
  - CT：`outputs/features/ct_radiomics_parotid.csv`、`outputs/features/ct_radiomics_submand.csv`
  - Dose：`outputs/features/dosiomics_parotid.csv`、`outputs/features/dosiomics_submand.csv`
- 单次三划分（three-way）split：`outputs/splits/threeway_{train,val,test}_ids.csv`
- 稳定特征（min3of5）：
  - CT：`outputs/selection/ct_ibsi_utest_lasso_seed42_k5/stable_features_min3of5.csv`
  - Dose：`outputs/selection/dose_ibsi_utest_lasso_seed42_k5/stable_features_min3of5.csv`

生成单次三划分 split（首次或想重置时执行）：
```bash
python make_splits.py --train-size 0.7 --val-size 0.2 --test-size 0.1
```

## 2. 常用命令（`xgb_train.py`）
单次三划分（读取 `outputs/splits/threeway_*_ids.csv`）：
```bash
python XGB/xgb_train.py --task ct_min3of5
python XGB/xgb_train.py --task dose_min3of5
python XGB/xgb_train.py --task ct_dose_early_min3of5
```

## 3. 对比与统计（`compare_svm_xgb.py`）
对比 SVM / XGB / LR 在单次三划分测试集预测上的 AUC（含 bootstrap CI）、以及两两比较的 DeLong / McNemar p-value：
```bash
python XGB/compare_svm_xgb.py
```

可选：输出“满足召回率/特异度约束”的阈值搜索结果（单次三划分）：
```bash
python XGB/compare_svm_xgb.py --clinical-thresholds --min-recall 0.75 --min-specificity 0.80 --objective youden
```

## 4. 输出产物
- 单次三划分：`XGB/omics/<task>/metric_auc/` 下生成 `test_predictions.csv`、`train_oof_predictions.csv`、`test_metrics.json`、`best_params.json` 等
- 跨模型对比（`compare_svm_xgb.py` 生成）：`outputs/model_performance_summary.csv`、`outputs/model_comparison_pvalues.csv`

## 5. 目录结构
- `xgb_train.py`：训练与评估主脚本（单次三划分）。
- `compare_svm_xgb.py`：跨模型指标汇总与统计对比工具。
- `omics/`：各任务输出目录。

---
**更新日期**：2025-12-29

# SVM 放射组学建模模块

本目录包含用于放射组学二分类的支持向量机（SVM）建模流水线，覆盖 **CT**、**Dose** 以及 **多模态早期融合（Early Fusion，特征拼接）**。

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

## 2. 常用命令（`svm_train.py`）

单次三划分（读取 `outputs/splits/threeway_*_ids.csv`）：
```bash
python SVM/svm_train.py --task ct_min3of5
python SVM/svm_train.py --task dose_min3of5
python SVM/svm_train.py --task ct_dose_early_min3of5_tuned
```

## 3. 汇总与表格（`make_table2.py`）
生成指标汇总表（默认输出到 `outputs/table2/table2_results.csv`，并会尝试生成 `outputs/table2/table2_pvalues.csv`）：
```bash
python SVM/make_table2.py --models ct_min3of5,dose_min3of5,ct_dose_early_min3of5_tuned
```

说明：
- `--models` 是逗号分隔的模型目录名，对应 `SVM/omics/<model>/metric_<metric>/`（默认 `metric_auc`）
- 本项目已移除外层五折训练流程；因此通常不会再生成 pooled 外层预测文件，也不会出现 `INTERNAL_CV_POOLED` 行

## 4. 输出产物
- 每个任务输出到 `SVM/omics/<task>/metric_auc/`，包含：
  - `train_oof_predictions.csv`
  - `test_predictions.csv`
  - `test_metrics.json`
  - `best_params.json`
  - `svm_bundle.joblib`
- 汇总表与显著性检验输出到：
  - `outputs/table2/table2_results.csv`
  - `outputs/table2/table2_pvalues.csv`（若存在可用的对比输入）

## 5. 目录结构
- `svm_train.py`：训练与预测主脚本（单次三划分）。
- `make_table2.py`：结果汇总与统计检验工具。
- `omics/`：各任务输出目录。

---
**更新日期**：2025-12-29

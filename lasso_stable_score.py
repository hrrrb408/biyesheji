import argparse
import hashlib
import json
import math
import os
import random
import re
import warnings
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from metrics_utils import decision_function_pos, predict_proba_pos


warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty=l1 with l1_ratio=0.0.*", category=UserWarning)


@dataclass(frozen=True)
class LassoConfig:
    cs: list[float]
    min_features: int
    max_features: int
    target_features: int
    max_iter: int


def _sha256_file(path: str) -> str | None:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_float_list(s: str) -> list[float]:
    toks = [t.strip() for t in str(s).split(",") if t.strip()]
    out: list[float] = []
    for t in toks:
        out.append(float(t))
    if not out:
        raise ValueError("Empty list")
    return out


def _format_float_for_path(x: float) -> str:
    s = f"{float(x):g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _load_drift_top_ks(*, path: str, split_key: str) -> dict[str, float]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        diag = json.load(f)
    drift = (diag or {}).get("drift") or {}
    split = drift.get(split_key) or {}
    top = split.get("top") or []
    out: dict[str, float] = {}
    for row in top:
        if not isinstance(row, dict):
            continue
        feat = row.get("feature")
        ks = row.get("ks_stat")
        if feat is None or ks is None:
            continue
        try:
            out[str(feat)] = float(ks)
        except Exception:
            continue
    return out


def _load_table(p: str) -> pd.DataFrame:
    ext = os.path.splitext(p)[1].lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    raise ValueError(f"Unsupported input: {p}")


def to_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def load_and_merge_inputs(paths: list[str], *, id_col: str, label_cols_to_drop: list[str]) -> pd.DataFrame:
    if not paths:
        raise ValueError("No input paths provided")
    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = _load_table(p)
        if id_col not in df.columns:
            raise ValueError(f"Missing id_col={id_col} in {p}")
        drop = [c for c in label_cols_to_drop if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        diag = [c for c in df.columns if str(c).lower().startswith("diagnostics_") or "diagnostics" in str(c).lower()]
        if diag:
            df = df.drop(columns=diag)

        non_id_cols = [c for c in df.columns if c != id_col]
        if non_id_cols:
            df[non_id_cols] = to_numeric_df(df[non_id_cols])

        dfs.append(df)
    out = dfs[0]
    for other in dfs[1:]:
        out = out.merge(other, on=id_col, how="inner")
    return out


def load_and_merge_inputs_with_prefix(
    paths: list[str],
    *,
    id_col: str,
    label_cols_to_drop: list[str],
    modality_tag: str,
) -> pd.DataFrame:
    if not paths:
        raise ValueError("No input paths provided")
    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = _load_table(p)
        if id_col not in df.columns:
            raise ValueError(f"Missing id_col={id_col} in {p}")
        drop = [c for c in label_cols_to_drop if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        diag = [c for c in df.columns if str(c).lower().startswith("diagnostics_") or "diagnostics" in str(c).lower()]
        if diag:
            df = df.drop(columns=diag)

        non_id_cols = [c for c in df.columns if c != id_col]
        if non_id_cols:
            df[non_id_cols] = to_numeric_df(df[non_id_cols])

        stem = os.path.splitext(os.path.basename(p))[0]
        organ = stem
        if modality_tag == "ct" and stem.startswith("ct_radiomics_"):
            organ = stem[len("ct_radiomics_") :]
        if modality_tag == "dose" and stem.startswith("dosiomics_"):
            organ = stem[len("dosiomics_") :]
        prefix = f"{modality_tag}__{organ}__"
        rename_map = {c: prefix + str(c) for c in df.columns if c != id_col}
        df = df.rename(columns=rename_map)
        dfs.append(df)

    out = dfs[0]
    for other in dfs[1:]:
        out = out.merge(other, on=id_col, how="inner")
    return out


def align_with_labels(
    df_feat: pd.DataFrame,
    *,
    labels_path: str,
    id_col: str,
    label_col_in_labels: str,
    label_col: str,
) -> pd.DataFrame:
    df_labels = _load_table(labels_path)
    if id_col not in df_labels.columns:
        raise ValueError(f"Missing id_col={id_col} in labels")
    if label_col_in_labels not in df_labels.columns:
        raise ValueError(f"Missing label_col_in_labels={label_col_in_labels} in labels")
    df_y = (
        df_labels[[id_col, label_col_in_labels]]
        .drop_duplicates(subset=[id_col])
        .rename(columns={label_col_in_labels: label_col})
    )
    df_y[label_col] = df_y[label_col].astype(int)
    df = df_feat.merge(df_y, on=id_col, how="inner")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    if not df[id_col].is_unique:
        raise ValueError("case_id is not unique after merge/label alignment")
    return df


def _load_ids_set(path: str, *, id_col: str) -> set[str]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    col = str(id_col) if str(id_col) in df.columns else str(df.columns[0])
    return set(df[col].astype(str).tolist())


def _split_by_threeway_ids(
    *,
    case_ids: np.ndarray,
    y: np.ndarray,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cid = np.asarray(case_ids, dtype=str)
    idx_all = np.arange(cid.shape[0], dtype=int)
    idx_train = idx_all[np.isin(cid, np.array(sorted(train_ids), dtype=str))]
    idx_val = idx_all[np.isin(cid, np.array(sorted(val_ids), dtype=str))]
    idx_test = idx_all[np.isin(cid, np.array(sorted(test_ids), dtype=str))]

    if idx_train.size == 0 or idx_val.size == 0 or idx_test.size == 0:
        raise ValueError(f"三划分样本为空：train={idx_train.size}, val={idx_val.size}, test={idx_test.size}")
    if np.unique(y[idx_train]).size < 2:
        raise ValueError("训练集不满足二分类")
    if np.unique(y[idx_val]).size < 2:
        raise ValueError("验证集不满足二分类")
    if np.unique(y[idx_test]).size < 2:
        raise ValueError("测试集不满足二分类")
    return (
        np.array(sorted(idx_train.tolist()), dtype=int),
        np.array(sorted(idx_val.tolist()), dtype=int),
        np.array(sorted(idx_test.tolist()), dtype=int),
    )


def _select_features_train_only(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    pipeline_kind: str,
    ibsi_rules: list[dict[str, str | bool]],
    extra_regexes: list[str],
    missing_rate_max: float,
    variance_min: float,
    corr_method: str,
    corr_threshold: float,
    u_test_p_threshold: float,
    u_test_fdr: bool,
    lasso_cfg: LassoConfig,
    lasso_penalty: str,
    lasso_l1_ratios: list[float],
    lasso_inner_cv: int,
    lasso_cv_select: str,
    lasso_min_nonzero: int,
    seed: int,
    outdir: str,
) -> tuple[list[str], dict[str, Any]]:
    X_tr = X_train.copy()
    X_va = X_val.copy()
    X_te = X_test.copy()

    removed_df = pd.DataFrame()
    if pipeline_kind != "legacy":
        X_tr, X_va, kept_s1, removed_df = _apply_ibsi_filter_by_train(
            X_tr,
            X_va,
            extra_regexes=extra_regexes,
            ibsi_rules=ibsi_rules,
        )
        X_te = X_te[[c for c in kept_s1 if c in X_te.columns]].copy()
        removed_df.to_csv(os.path.join(outdir, "step1_ibsi_removed_features.csv"), index=False)
        pd.DataFrame({"feature": kept_s1}).to_csv(os.path.join(outdir, "step1_ibsi_kept_features.txt"), index=False, header=False)

        X_tr = X_tr[kept_s1].copy()
        X_va = X_va[kept_s1].copy()

    X_tr, X_va, kept_s_flt, meta_filter = _filter_columns_by_train_stats(
        X_tr,
        X_va,
        missing_rate_max=float(missing_rate_max),
        variance_min=float(variance_min),
        corr_method=str(corr_method),
        corr_threshold=float(corr_threshold),
    )
    X_te = X_te[kept_s_flt].copy()
    with open(os.path.join(outdir, "step0_train_stats_filter_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_filter, f, ensure_ascii=False, indent=2)

    kept_s2, stats_df, meta_ut = _u_test_prefilter_train_only(
        X_tr,
        y_train,
        p_threshold=float(u_test_p_threshold),
        compute_fdr=bool(u_test_fdr),
    )
    stats_df.to_csv(os.path.join(outdir, "step2_u_test_stats.csv"), index=False)
    pd.DataFrame({"feature": kept_s2}).to_csv(os.path.join(outdir, "step2_u_test_kept_features.txt"), index=False, header=False)

    if not kept_s2:
        raise ValueError("U检验筛选后特征为空，请放宽阈值或检查数据")

    X_tr_s2 = X_tr[kept_s2].copy()
    cv_df = _auc_cv_over_grid(
        X_tr_s2,
        y_train,
        penalty=str(lasso_penalty),
        cs=lasso_cfg.cs,
        l1_ratios=lasso_l1_ratios,
        folds=int(lasso_inner_cv),
        seed=int(seed),
        max_iter=int(lasso_cfg.max_iter),
    )
    cv_df.to_csv(os.path.join(outdir, "step3_lasso_cv_curve.csv"), index=False)
    if str(lasso_cv_select) == "best_auc":
        row = cv_df.sort_values(["mean_auc", "C", "l1_ratio"], ascending=[False, True, True]).iloc[0]
        chosen_C = float(row["C"])
        chosen_l1_ratio = float(row.get("l1_ratio", 1.0))
    else:
        chosen_C, chosen_l1_ratio = _choose_grid_by_1se(cv_df)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_tr_s2))
    if str(lasso_penalty) == "elasticnet":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(chosen_l1_ratio),
            max_iter=int(lasso_cfg.max_iter),
            class_weight="balanced",
            C=float(chosen_C),
            random_state=int(seed),
        )
    else:
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=int(lasso_cfg.max_iter),
            class_weight="balanced",
            C=float(chosen_C),
            random_state=int(seed),
        )
    clf.fit(X_tr_sc, y_train)
    coef = np.ravel(clf.coef_).astype(np.float64)
    abs_coef = np.abs(coef)
    nz = np.where(abs_coef > 1e-12)[0]
    if nz.size == 0:
        chosen_idx = np.argsort(-abs_coef)[: max(int(lasso_min_nonzero), 1)]
    else:
        chosen_idx = nz
        if chosen_idx.size < int(lasso_min_nonzero):
            chosen_idx = np.argsort(-abs_coef)[: int(lasso_min_nonzero)]
    chosen_idx = np.array(sorted(set(chosen_idx.tolist())), dtype=int)
    selected = [kept_s2[i] for i in chosen_idx.tolist()]
    pd.DataFrame({"feature": selected}).to_csv(os.path.join(outdir, "step3_lasso_selected_features.txt"), index=False, header=False)
    pd.DataFrame({"feature": selected, "abs_coef": abs_coef[chosen_idx].tolist()}).to_csv(
        os.path.join(outdir, "step3_lasso_coefficients.csv"),
        index=False,
    )

    meta = {
        "step1_removed": int(removed_df["feature"].nunique()) if not removed_df.empty else 0,
        "step0_train_stats_filter": meta_filter,
        "step2_u_test": meta_ut,
        "lasso": {
            "penalty": str(lasso_penalty),
            "l1_ratio": float(chosen_l1_ratio),
            "C": float(chosen_C),
            "cv_select": str(lasso_cv_select),
            "inner_cv": int(lasso_inner_cv),
            "min_nonzero": int(lasso_min_nonzero),
            "n_selected": int(len(selected)),
        },
    }
    with open(os.path.join(outdir, "feature_selection_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return selected, meta


def _fit_and_eval_models(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int,
    outdir: str,
) -> dict[str, Any]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imputer.fit_transform(X_train))
    X_va = scaler.transform(imputer.transform(X_val))
    X_te = scaler.transform(imputer.transform(X_test))

    results: dict[str, Any] = {}

    lr = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=5000,
        random_state=int(seed),
    )
    lr.fit(X_tr, y_train)
    lr_val = predict_proba_pos(lr, X_va).astype(float)
    lr_test = predict_proba_pos(lr, X_te).astype(float)
    results["lr"] = {
        "val_auc": float(roc_auc_score(y_val, lr_val)),
        "test_auc": float(roc_auc_score(y_test, lr_test)),
    }
    joblib.dump({"imputer": imputer, "scaler": scaler, "model": lr}, os.path.join(outdir, "model_lr.joblib"))

    svm_base = LinearSVC(
        C=1.0,
        class_weight="balanced",
        dual=False,
        max_iter=500000,
        random_state=int(seed),
    )
    svm = CalibratedClassifierCV(svm_base, cv=3, method="sigmoid")
    svm.fit(X_tr, y_train)
    svm_val = predict_proba_pos(svm, X_va).astype(float)
    svm_test = predict_proba_pos(svm, X_te).astype(float)
    results["svm"] = {
        "val_auc": float(roc_auc_score(y_val, svm_val)),
        "test_auc": float(roc_auc_score(y_test, svm_test)),
    }
    joblib.dump({"imputer": imputer, "scaler": scaler, "model": svm}, os.path.join(outdir, "model_svm.joblib"))

    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError("需要安装 xgboost 才能训练 XGB 模型") from e

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=1,
        random_state=int(seed),
    )
    xgb.fit(X_tr, y_train, verbose=False)
    xgb_val = predict_proba_pos(xgb, X_va).astype(float)
    xgb_test = predict_proba_pos(xgb, X_te).astype(float)
    results["xgb"] = {
        "val_auc": float(roc_auc_score(y_val, xgb_val)),
        "test_auc": float(roc_auc_score(y_test, xgb_test)),
    }
    joblib.dump({"imputer": imputer, "scaler": scaler, "model": xgb}, os.path.join(outdir, "model_xgb.joblib"))

    pd.DataFrame(
        {
            "split": ["val"] * int(len(y_val)) + ["test"] * int(len(y_test)),
            "y": np.concatenate([y_val, y_test]).astype(int).tolist(),
            "lr_proba": np.concatenate([lr_val, lr_test]).astype(float).tolist(),
            "svm_proba": np.concatenate([svm_val, svm_test]).astype(float).tolist(),
            "xgb_proba": np.concatenate([xgb_val, xgb_test]).astype(float).tolist(),
        }
    ).to_csv(os.path.join(outdir, "predictions_val_test.csv"), index=False)

    with open(os.path.join(outdir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def _filter_columns_by_train_stats(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    missing_rate_max: float,
    variance_min: float,
    corr_method: str,
    corr_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    missing_rate = X_train.isna().mean()
    keep_cols = missing_rate[missing_rate <= float(missing_rate_max)].index.tolist()
    X_train = X_train[keep_cols]
    X_test = X_test[keep_cols]

    variances = X_train.var(axis=0, skipna=True)
    keep_cols2 = variances[(variances > float(variance_min)) & (~variances.isna())].index.tolist()
    X_train = X_train[keep_cols2]
    X_test = X_test[keep_cols2]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train).astype(np.float64, copy=False)
    if corr_method == "spearman":
        X_corr_base = pd.DataFrame(X_train_imp, columns=X_train.columns).rank(method="average").to_numpy(dtype=np.float64)
    else:
        X_corr_base = X_train_imp

    means = X_corr_base.mean(axis=0)
    stds = X_corr_base.std(axis=0)
    keep_nonconst = stds > 0.0
    kept_cols_nonconst = np.array(X_train.columns.tolist(), dtype=object)[keep_nonconst].tolist()
    X_corr_base = X_corr_base[:, keep_nonconst]
    means = means[keep_nonconst]
    stds = stds[keep_nonconst]

    Xz = (X_corr_base - means) / stds
    denom = max(int(Xz.shape[0] - 1), 1)
    corr_abs = np.abs((Xz.T @ Xz) / float(denom))
    np.fill_diagonal(corr_abs, 0.0)
    upper = np.triu(corr_abs, k=1)
    to_drop_mask = (upper > float(corr_threshold)).any(axis=0)
    to_drop = np.array(kept_cols_nonconst, dtype=object)[to_drop_mask].tolist()
    keep_cols3 = [c for c in kept_cols_nonconst if c not in set(to_drop)]
    X_train = X_train[keep_cols3]
    X_test = X_test[keep_cols3]

    meta = {
        "n_input": int(len(missing_rate.index)),
        "n_after_missing": int(len(keep_cols)),
        "n_after_variance": int(len(keep_cols2)),
        "n_after_corr": int(len(keep_cols3)),
        "dropped_constant": int(len(keep_cols2) - int(np.sum(keep_nonconst))),
        "dropped_corr": int(len(to_drop)),
    }
    return X_train, X_test, keep_cols3, meta


def _load_ibsi_exclude_rules(path: str) -> list[dict[str, str | bool]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rules: list[dict[str, str | bool]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                rules.append({"pattern": item, "regex": True, "reason": "ibsi_exclude"})
            elif isinstance(item, dict):
                pattern = str(item.get("pattern") or item.get("name") or "").strip()
                if not pattern:
                    continue
                rules.append(
                    {
                        "pattern": pattern,
                        "regex": bool(item.get("regex", False)),
                        "reason": str(item.get("reason") or "ibsi_exclude"),
                    }
                )
    elif isinstance(data, dict):
        items = data.get("rules")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    rules.append({"pattern": item, "regex": True, "reason": "ibsi_exclude"})
                elif isinstance(item, dict):
                    pattern = str(item.get("pattern") or item.get("name") or "").strip()
                    if not pattern:
                        continue
                    rules.append(
                        {
                            "pattern": pattern,
                            "regex": bool(item.get("regex", False)),
                            "reason": str(item.get("reason") or "ibsi_exclude"),
                        }
                    )
    return rules


def _apply_ibsi_filter_by_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    extra_regexes: list[str],
    ibsi_rules: list[dict[str, str | bool]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    removed: list[dict[str, str]] = []
    colset = set(X_train.columns.tolist())
    to_drop: set[str] = set()

    for rgx in extra_regexes:
        try:
            pat = re.compile(rgx)
        except re.error:
            continue
        for c in colset:
            if pat.search(str(c)):
                to_drop.add(str(c))
                removed.append({"feature": str(c), "reason": f"regex:{rgx}"})

    for rule in ibsi_rules:
        pattern = str(rule.get("pattern") or "").strip()
        if not pattern:
            continue
        is_regex = bool(rule.get("regex", False))
        reason = str(rule.get("reason") or "ibsi_exclude")
        if is_regex:
            try:
                pat = re.compile(pattern)
            except re.error:
                continue
            for c in colset:
                if pat.search(str(c)):
                    to_drop.add(str(c))
                    removed.append({"feature": str(c), "reason": reason})
        else:
            if pattern in colset:
                to_drop.add(pattern)
                removed.append({"feature": pattern, "reason": reason})

    keep_cols = [c for c in X_train.columns.tolist() if c not in to_drop]
    X_train2 = X_train[keep_cols].copy()
    X_test2 = X_test[[c for c in keep_cols if c in X_test.columns]].copy()
    removed_df = pd.DataFrame(removed).drop_duplicates(subset=["feature"], keep="first")
    return X_train2, X_test2, keep_cols, removed_df


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    n = int(p.size)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1, dtype=np.int64)
    q = p * float(n) / ranks.astype(np.float64)
    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q_sorted, 0.0, 1.0)
    return out


def _u_test_prefilter_train_only(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    p_threshold: float,
    compute_fdr: bool,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    try:
        from scipy.stats import mannwhitneyu
    except Exception as e:
        raise RuntimeError("scipy is required for Mann–Whitney U test") from e

    y_train = np.asarray(y_train, dtype=int)
    if np.unique(y_train).size != 2:
        raise ValueError("y_train must contain both classes 0 and 1")

    missing_all = X_train.isna().all(axis=0)
    cols0 = missing_all[missing_all == False].index.tolist()
    X0 = X_train[cols0].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X0).astype(np.float64, copy=False)
    var = np.var(X_imp, axis=0)
    keep_var = var > 0.0
    cols1 = np.array(cols0, dtype=object)[keep_var].tolist()
    X_imp2 = X_imp[:, keep_var]

    g0 = y_train == 0
    g1 = y_train == 1
    if int(np.sum(g0)) == 0 or int(np.sum(g1)) == 0:
        raise ValueError("Both classes must be present in y_train")

    med0 = np.median(X_imp2[g0, :], axis=0)
    med1 = np.median(X_imp2[g1, :], axis=0)
    effect = (med1 - med0).astype(np.float64, copy=False)

    pvals = np.empty(X_imp2.shape[1], dtype=np.float64)
    for j in range(X_imp2.shape[1]):
        x0 = X_imp2[g0, j]
        x1 = X_imp2[g1, j]
        try:
            res = mannwhitneyu(x0, x1, alternative="two-sided", method="asymptotic")
            pvals[j] = float(res.pvalue)
        except Exception:
            pvals[j] = 1.0

    qvals = _bh_fdr(pvals) if compute_fdr else None
    stats = {
        "feature": cols1,
        "p_value": pvals.tolist(),
        "effect_median_diff": effect.tolist(),
    }
    if qvals is not None:
        stats["q_value"] = qvals.tolist()
    stats_df = pd.DataFrame(stats).sort_values(["p_value", "feature"], ascending=[True, True]).reset_index(drop=True)
    kept = stats_df[stats_df["p_value"].astype(float) < float(p_threshold)]["feature"].astype(str).tolist()
    meta = {
        "n_input": int(X_train.shape[1]),
        "n_after_all_missing_drop": int(len(cols0)),
        "n_after_constant_drop": int(len(cols1)),
        "p_threshold": float(p_threshold),
        "compute_fdr": bool(compute_fdr),
        "n_kept": int(len(kept)),
    }
    return kept, stats_df, meta


def _auc_cv_over_C(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cs: list[float],
    folds: int,
    seed: int,
    max_iter: int,
) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    rows: list[dict[str, Any]] = []
    for C in cs:
        aucs: list[float] = []
        for tr, te in skf.split(np.zeros(shape=(len(y), 1)), y):
            X_tr = X.iloc[tr]
            X_te = X.iloc[te]
            y_tr = y[tr]
            y_te = y[te]

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_tr))
            X_te_sc = scaler.transform(imputer.transform(X_te))

            clf = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=int(max_iter),
                class_weight="balanced",
                C=float(C),
                random_state=int(seed),
            )
            clf.fit(X_tr_sc, y_tr)
            proba = predict_proba_pos(clf, X_te_sc).astype(float)
            aucs.append(float(roc_auc_score(y_te, proba)))

        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        se_auc = float(std_auc / math.sqrt(max(len(aucs), 1)))
        rows.append({"C": float(C), "mean_auc": mean_auc, "std_auc": std_auc, "se_auc": se_auc})
    return pd.DataFrame(rows).sort_values(["C"], ascending=[True]).reset_index(drop=True)


def _choose_C_by_1se(cv_df: pd.DataFrame) -> float:
    df = cv_df.dropna(subset=["mean_auc"]).copy()
    if df.empty:
        return float(cv_df["C"].iloc[0])
    best_mean = float(df["mean_auc"].max())
    best_row = df.sort_values(["mean_auc", "C"], ascending=[False, True]).iloc[0]
    threshold = best_mean - float(best_row.get("se_auc", 0.0))
    ok = df[df["mean_auc"].astype(float) >= float(threshold)].sort_values(["C"], ascending=[True])
    if ok.empty:
        return float(best_row["C"])
    return float(ok.iloc[0]["C"])


def _auc_cv_over_grid(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    penalty: str,
    cs: list[float],
    l1_ratios: list[float],
    folds: int,
    seed: int,
    max_iter: int,
) -> pd.DataFrame:
    penalty = str(penalty).strip().lower()
    if penalty not in ("l1", "elasticnet"):
        raise ValueError(f"Unsupported lasso_penalty: {penalty}")
    if penalty == "l1":
        base = _auc_cv_over_C(X, y, cs=cs, folds=folds, seed=seed, max_iter=max_iter)
        base.insert(0, "l1_ratio", 1.0)
        base.insert(0, "penalty", "l1")
        return base

    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    rows: list[dict[str, Any]] = []
    for l1r in l1_ratios:
        l1r_f = float(l1r)
        for C in cs:
            aucs: list[float] = []
            for tr, te in skf.split(np.zeros(shape=(len(y), 1)), y):
                X_tr = X.iloc[tr]
                X_te = X.iloc[te]
                y_tr = y[tr]
                y_te = y[te]

                imputer = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_tr))
                X_te_sc = scaler.transform(imputer.transform(X_te))

                clf = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=float(l1r_f),
                    max_iter=int(max_iter),
                    class_weight="balanced",
                    C=float(C),
                    random_state=int(seed),
                )
                clf.fit(X_tr_sc, y_tr)
                proba = predict_proba_pos(clf, X_te_sc).astype(float)
                aucs.append(float(roc_auc_score(y_te, proba)))

            mean_auc = float(np.mean(aucs)) if aucs else float("nan")
            std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            se_auc = float(std_auc / math.sqrt(max(len(aucs), 1)))
            rows.append(
                {
                    "penalty": "elasticnet",
                    "l1_ratio": float(l1r_f),
                    "C": float(C),
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                    "se_auc": se_auc,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["l1_ratio", "C"], ascending=[True, True])
        .reset_index(drop=True)
    )


def _choose_grid_by_1se(cv_df: pd.DataFrame) -> tuple[float, float]:
    df = cv_df.dropna(subset=["mean_auc"]).copy()
    if df.empty:
        row0 = cv_df.iloc[0]
        return float(row0["C"]), float(row0.get("l1_ratio", 1.0))

    df["C"] = df["C"].astype(float)
    if "l1_ratio" not in df.columns:
        df["l1_ratio"] = 1.0
    df["l1_ratio"] = df["l1_ratio"].astype(float)

    best_row = df.sort_values(["mean_auc", "C", "l1_ratio"], ascending=[False, True, True]).iloc[0]
    best_mean = float(best_row["mean_auc"])
    threshold = best_mean - float(best_row.get("se_auc", 0.0))
    ok = df[df["mean_auc"].astype(float) >= float(threshold)].sort_values(["C", "l1_ratio"], ascending=[True, True])
    if ok.empty:
        return float(best_row["C"]), float(best_row["l1_ratio"])
    row = ok.iloc[0]
    return float(row["C"]), float(row["l1_ratio"])


def _select_by_l1_logistic(
    *,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    feature_names: list[str],
    cfg: LassoConfig,
    seed: int,
) -> tuple[np.ndarray, list[str], list[float], dict[str, Any], LogisticRegression]:
    min_features = int(cfg.min_features)
    max_features = int(cfg.max_features)
    target_features = int(cfg.target_features)
    if min_features < 1:
        min_features = 1
    if max_features < min_features:
        max_features = min_features
    if target_features < min_features:
        target_features = min_features
    if target_features > max_features:
        target_features = max_features

    best: dict[str, Any] | None = None
    best_clf: LogisticRegression | None = None
    for C in cfg.cs:
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=int(cfg.max_iter),
            class_weight="balanced",
            C=float(C),
            random_state=int(seed),
        )
        clf.fit(X_tr, y_tr)
        coef = np.ravel(clf.coef_).astype(np.float64)
        abs_coef = np.abs(coef)
        nz = np.where(abs_coef > 1e-12)[0]
        if nz.size == 0:
            order = np.argsort(-abs_coef)
            chosen = order[:min_features]
        else:
            chosen = nz
        if chosen.size > max_features:
            order = chosen[np.argsort(-abs_coef[chosen])]
            chosen = order[:max_features]
        if chosen.size < min_features:
            order = np.argsort(-abs_coef)
            chosen = order[:min_features]

        score = abs(int(chosen.size) - int(target_features))
        if best is None or score < int(best["score"]):
            best = {
                "C": float(C),
                "score": int(score),
                "n_selected": int(chosen.size),
                "indices": chosen.astype(int),
                "abs_coef": abs_coef,
            }
            best_clf = clf
            if int(best["score"]) == 0:
                break

    assert best is not None and best_clf is not None
    idx = best["indices"]
    selected_names = [feature_names[i] for i in idx.tolist()]
    selected_abscoef = best["abs_coef"][idx].tolist()
    meta = {
        "C": float(best["C"]),
        "n_selected": int(best["n_selected"]),
        "min_features": int(min_features),
        "max_features": int(max_features),
        "target_features": int(target_features),
    }
    return idx, selected_names, selected_abscoef, meta, best_clf


def _split_indices(
    *,
    case_ids: np.ndarray,
    y: np.ndarray,
    folds: int,
    seed: int,
    use_fold_assignments: bool,
    fold_assignments_path: str,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if use_fold_assignments and os.path.exists(fold_assignments_path):
        fa = pd.read_csv(fold_assignments_path)
        if "case_id" not in fa.columns or "fold_id" not in fa.columns:
            raise ValueError("fold_assignments must contain case_id and fold_id")
        fold_map = dict(zip(fa["case_id"].astype(str).tolist(), fa["fold_id"].astype(int).tolist()))
        fold_id = np.array([fold_map.get(str(cid), -1) for cid in case_ids], dtype=int)
        keep = fold_id != -1
        idx_keep = np.where(keep)[0]
        fold_id2 = fold_id[keep]
        out: list[tuple[int, np.ndarray, np.ndarray]] = []
        for fold in range(1, int(folds) + 1):
            idx_test2 = idx_keep[fold_id2 == int(fold)]
            idx_train2 = idx_keep[fold_id2 != int(fold)]
            if idx_test2.size == 0 or idx_train2.size == 0:
                continue
            if np.unique(y[idx_train2]).size < 2 or np.unique(y[idx_test2]).size < 2:
                continue
            out.append((int(fold), np.array(sorted(idx_train2.tolist())), np.array(sorted(idx_test2.tolist()))))
        if out:
            return out

    skf = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    out2: list[tuple[int, np.ndarray, np.ndarray]] = []
    for k, (tr, te) in enumerate(skf.split(np.zeros(shape=(len(case_ids), 1)), y), start=1):
        out2.append((int(k), np.array(sorted(tr.tolist())), np.array(sorted(te.tolist()))))
    return out2


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pipeline",
        choices=["legacy", "ibsi_utest_lasso", "threeway_trainvaltest"],
        default="ibsi_utest_lasso",
    )
    ap.add_argument("--modality", choices=["ct", "dose", "both"], required=True)
    ap.add_argument("--inputs", type=str, default="")
    ap.add_argument("--ct-inputs", type=str, default="")
    ap.add_argument("--dose-inputs", type=str, default="")
    ap.add_argument("--labels", default="outputs/curation/index_with_labels.csv")
    ap.add_argument("--id-col", default="case_id")
    ap.add_argument("--label-col-in-labels", default="y")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-fold-assignments", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fold-assignments-path", default="outputs/fold_assignments_seed42_k5.csv")
    # three-queue args removed
    ap.add_argument("--train-ids-path", default="outputs/splits/threeway_train_ids.csv")
    ap.add_argument("--val-ids-path", default="outputs/splits/threeway_val_ids.csv")
    ap.add_argument("--test-ids-path", default="outputs/splits/threeway_test_ids.csv")

    ap.add_argument("--ibsi-exclude-json", type=str, default="")
    ap.add_argument("--ibsi-extra-regex", type=str, default="(?i)^diagnostics_")

    ap.add_argument("--u-test-p-threshold", type=float, default=0.05)
    ap.add_argument("--u-test-fdr", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--missing-rate-max", type=float, default=0.3)
    ap.add_argument("--variance-min", type=float, default=1e-8)
    ap.add_argument("--restrict-ids-path", type=str, default="", help="Path to a CSV containing case_ids to keep (e.g., for train-only selection).")
    ap.add_argument("--corr-method", default="spearman", choices=["pearson", "spearman"])
    ap.add_argument("--corr-threshold", type=float, default=0.95)

    ap.add_argument("--lasso-cs", type=str, default="0.01,0.03,0.1,0.3,1,3,10")
    ap.add_argument("--lasso-penalty", choices=["l1", "elasticnet"], default="l1")
    ap.add_argument("--lasso-l1-ratios", type=str, default="0.1,0.3,0.5")
    ap.add_argument("--lasso-min-features", type=int, default=30)
    ap.add_argument("--lasso-max-features", type=int, default=200)
    ap.add_argument("--lasso-target-features", type=int, default=80)
    ap.add_argument("--lasso-max-iter", type=int, default=2000)
    ap.add_argument("--lasso-inner-cv", type=int, default=5)
    ap.add_argument("--lasso-cv-select", choices=["best_auc", "1se"], default="1se")
    ap.add_argument("--lasso-min-nonzero", type=int, default=1)
    ap.add_argument("--stable-min-folds", type=int, default=3)
    ap.add_argument("--drift-diagnostics-path", type=str, default="")
    ap.add_argument("--drift-split-key", choices=["train_vs_test", "train_vs_val"], default="train_vs_test")
    ap.add_argument("--drift-ks-max", type=float, default=-1.0)

    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args(argv)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    if str(args.pipeline) == "threeway_trainvaltest":
        input_paths = []
    else:
        if str(args.inputs).strip():
            input_paths = [p.strip() for p in str(args.inputs).split(",") if p.strip()]
        else:
            if args.modality == "ct":
                input_paths = [
                    "outputs/features/ct_radiomics_parotid.csv",
                    "outputs/features/ct_radiomics_submand.csv",
                ]
            else:
                input_paths = [
                    "outputs/features/dosiomics_parotid.csv",
                    "outputs/features/dosiomics_submand.csv",
                ]

    outdir = str(args.outdir).strip()
    if not outdir:
        if str(args.pipeline) == "threeway_trainvaltest":
            outdir = os.path.join(
                "outputs",
                "threeway",
                f"{args.modality}_trainvaltest_seed{int(args.seed)}",
            )
        elif str(args.pipeline) == "legacy":
            outdir = os.path.join(
                "outputs",
                "exp",
                "singleomics_5fold_seed42",
                f"lasso_{args.modality}_stable",
            )
        else:
            outdir = os.path.join(
                "outputs",
                "selection",
                f"{args.modality}_ibsi_utest_lasso_seed{int(args.seed)}_k{int(args.folds)}",
            )
            if args.restrict_ids_path:
                outdir += "_trainonly"
    os.makedirs(outdir, exist_ok=True)
    prev_run_meta_path = os.path.join(outdir, "run_meta.json")
    prev_run_meta: dict[str, Any] | None = None
    if os.path.exists(prev_run_meta_path):
        try:
            with open(prev_run_meta_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                prev_run_meta = loaded
        except Exception:
            prev_run_meta = None

    if str(args.pipeline) == "threeway_trainvaltest":
        if str(args.modality) == "ct":
            ct_paths = [p.strip() for p in str(args.ct_inputs or args.inputs).split(",") if p.strip()] if str(args.ct_inputs or args.inputs).strip() else [
                "outputs/features/ct_radiomics_parotid.csv",
                "outputs/features/ct_radiomics_submand.csv",
            ]
            df_feat = load_and_merge_inputs_with_prefix(
                ct_paths,
                id_col=str(args.id_col),
                label_cols_to_drop=[str(args.label_col_in_labels), str(args.label_col)],
                modality_tag="ct",
            )
        elif str(args.modality) == "dose":
            dose_paths = [p.strip() for p in str(args.dose_inputs or args.inputs).split(",") if p.strip()] if str(args.dose_inputs or args.inputs).strip() else [
                "outputs/features/dosiomics_parotid.csv",
                "outputs/features/dosiomics_submand.csv",
            ]
            df_feat = load_and_merge_inputs_with_prefix(
                dose_paths,
                id_col=str(args.id_col),
                label_cols_to_drop=[str(args.label_col_in_labels), str(args.label_col)],
                modality_tag="dose",
            )
        else:
            ct_paths = [p.strip() for p in str(args.ct_inputs).split(",") if p.strip()] if str(args.ct_inputs).strip() else [
                "outputs/features/ct_radiomics_parotid.csv",
                "outputs/features/ct_radiomics_submand.csv",
            ]
            dose_paths = [p.strip() for p in str(args.dose_inputs).split(",") if p.strip()] if str(args.dose_inputs).strip() else [
                "outputs/features/dosiomics_parotid.csv",
                "outputs/features/dosiomics_submand.csv",
            ]
            df_ct = load_and_merge_inputs_with_prefix(
                ct_paths,
                id_col=str(args.id_col),
                label_cols_to_drop=[str(args.label_col_in_labels), str(args.label_col)],
                modality_tag="ct",
            )
            df_dose = load_and_merge_inputs_with_prefix(
                dose_paths,
                id_col=str(args.id_col),
                label_cols_to_drop=[str(args.label_col_in_labels), str(args.label_col)],
                modality_tag="dose",
            )
            df_feat = df_ct.merge(df_dose, on=str(args.id_col), how="inner")
    else:
        df_feat = load_and_merge_inputs(
            input_paths,
            id_col=str(args.id_col),
            label_cols_to_drop=[str(args.label_col_in_labels), str(args.label_col)],
        )
    df = align_with_labels(
        df_feat,
        labels_path=str(args.labels),
        id_col=str(args.id_col),
        label_col_in_labels=str(args.label_col_in_labels),
        label_col=str(args.label_col),
    )

    if str(args.pipeline) != "threeway_trainvaltest" and args.restrict_ids_path and os.path.exists(args.restrict_ids_path):
        restrict_df = pd.read_csv(args.restrict_ids_path)
        # Assume 'case_id' column exists, or use the first column
        restrict_col = str(args.id_col) if str(args.id_col) in restrict_df.columns else restrict_df.columns[0]
        restrict_ids = set(restrict_df[restrict_col].astype(str).values)
        
        original_count = len(df)
        df = df[df[str(args.id_col)].astype(str).isin(restrict_ids)].copy()
        print(f"[INFO] Restricted samples from {original_count} to {len(df)} using IDs from {args.restrict_ids_path}")

    case_ids = df[str(args.id_col)].astype(str).values
    y = df[str(args.label_col)].astype(int).values
    X_df = df.drop(columns=[str(args.label_col)])
    X_df = X_df.drop(columns=[str(args.id_col)])

    if str(args.pipeline) == "threeway_trainvaltest":
        train_ids = _load_ids_set(str(args.train_ids_path), id_col=str(args.id_col))
        val_ids = _load_ids_set(str(args.val_ids_path), id_col=str(args.id_col))
        test_ids = _load_ids_set(str(args.test_ids_path), id_col=str(args.id_col))
        idx_train, idx_val, idx_test = _split_by_threeway_ids(
            case_ids=case_ids,
            y=y,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )

        ibsi_rules = _load_ibsi_exclude_rules(str(args.ibsi_exclude_json))
        extra_regexes = [r.strip() for r in str(args.ibsi_extra_regex).split(",") if r.strip()]
        ibsi_sha = _sha256_file(str(args.ibsi_exclude_json))

        lasso_cfg = LassoConfig(
            cs=_parse_float_list(str(args.lasso_cs)),
            min_features=int(args.lasso_min_features),
            max_features=int(args.lasso_max_features),
            target_features=int(args.lasso_target_features),
            max_iter=int(args.lasso_max_iter),
        )
        lasso_penalty = str(args.lasso_penalty).strip().lower()
        lasso_l1_ratios = _parse_float_list(str(args.lasso_l1_ratios))

        X_train_all = X_df.iloc[idx_train].copy()
        X_val_all = X_df.iloc[idx_val].copy()
        X_test_all = X_df.iloc[idx_test].copy()
        y_train = y[idx_train]
        y_val = y[idx_val]
        y_test = y[idx_test]

        if str(args.modality) == "both":
            ct_cols = [c for c in X_train_all.columns.tolist() if str(c).startswith("ct__")]
            dose_cols = [c for c in X_train_all.columns.tolist() if str(c).startswith("dose__")]
            if not ct_cols or not dose_cols:
                raise ValueError(f"both 模式需要 ct__ 与 dose__ 前缀列：ct={len(ct_cols)}, dose={len(dose_cols)}")

            ct_dir = os.path.join(outdir, "ct")
            dose_dir = os.path.join(outdir, "dose")
            os.makedirs(ct_dir, exist_ok=True)
            os.makedirs(dose_dir, exist_ok=True)

            ct_selected, ct_meta = _select_features_train_only(
                X_train=X_train_all[ct_cols],
                y_train=y_train,
                X_val=X_val_all[ct_cols],
                X_test=X_test_all[ct_cols],
                pipeline_kind="ibsi_utest_lasso",
                ibsi_rules=ibsi_rules,
                extra_regexes=extra_regexes,
                missing_rate_max=float(args.missing_rate_max),
                variance_min=float(args.variance_min),
                corr_method=str(args.corr_method),
                corr_threshold=float(args.corr_threshold),
                u_test_p_threshold=float(args.u_test_p_threshold),
                u_test_fdr=bool(args.u_test_fdr),
                lasso_cfg=lasso_cfg,
                lasso_penalty=lasso_penalty,
                lasso_l1_ratios=lasso_l1_ratios,
                lasso_inner_cv=int(args.lasso_inner_cv),
                lasso_cv_select=str(args.lasso_cv_select),
                lasso_min_nonzero=int(args.lasso_min_nonzero),
                seed=int(args.seed) + 101,
                outdir=ct_dir,
            )
            dose_selected, dose_meta = _select_features_train_only(
                X_train=X_train_all[dose_cols],
                y_train=y_train,
                X_val=X_val_all[dose_cols],
                X_test=X_test_all[dose_cols],
                pipeline_kind="ibsi_utest_lasso",
                ibsi_rules=ibsi_rules,
                extra_regexes=extra_regexes,
                missing_rate_max=float(args.missing_rate_max),
                variance_min=float(args.variance_min),
                corr_method=str(args.corr_method),
                corr_threshold=float(args.corr_threshold),
                u_test_p_threshold=float(args.u_test_p_threshold),
                u_test_fdr=bool(args.u_test_fdr),
                lasso_cfg=lasso_cfg,
                lasso_penalty=lasso_penalty,
                lasso_l1_ratios=lasso_l1_ratios,
                lasso_inner_cv=int(args.lasso_inner_cv),
                lasso_cv_select=str(args.lasso_cv_select),
                lasso_min_nonzero=int(args.lasso_min_nonzero),
                seed=int(args.seed) + 202,
                outdir=dose_dir,
            )
            selected = ct_selected + dose_selected
            pd.DataFrame({"feature": selected}).to_csv(os.path.join(outdir, "selected_features_all.csv"), index=False)
            metrics = _fit_and_eval_models(
                X_train=X_train_all[selected],
                y_train=y_train,
                X_val=X_val_all[selected],
                y_val=y_val,
                X_test=X_test_all[selected],
                y_test=y_test,
                seed=int(args.seed),
                outdir=outdir,
            )
            meta = {
                "pipeline": "threeway_trainvaltest",
                "modality": "both",
                "labels": str(args.labels),
                "train_ids_path": str(args.train_ids_path),
                "val_ids_path": str(args.val_ids_path),
                "test_ids_path": str(args.test_ids_path),
                "n_train": int(idx_train.size),
                "n_val": int(idx_val.size),
                "n_test": int(idx_test.size),
                "ibsi_exclude_json": str(args.ibsi_exclude_json),
                "ibsi_exclude_sha256": ibsi_sha,
                "ct": {"n_selected": int(len(ct_selected)), "meta_path": os.path.join("ct", "feature_selection_meta.json")},
                "dose": {"n_selected": int(len(dose_selected)), "meta_path": os.path.join("dose", "feature_selection_meta.json")},
                "models": metrics,
            }
        else:
            selected, sel_meta = _select_features_train_only(
                X_train=X_train_all,
                y_train=y_train,
                X_val=X_val_all,
                X_test=X_test_all,
                pipeline_kind="ibsi_utest_lasso",
                ibsi_rules=ibsi_rules,
                extra_regexes=extra_regexes,
                missing_rate_max=float(args.missing_rate_max),
                variance_min=float(args.variance_min),
                corr_method=str(args.corr_method),
                corr_threshold=float(args.corr_threshold),
                u_test_p_threshold=float(args.u_test_p_threshold),
                u_test_fdr=bool(args.u_test_fdr),
                lasso_cfg=lasso_cfg,
                lasso_penalty=lasso_penalty,
                lasso_l1_ratios=lasso_l1_ratios,
                lasso_inner_cv=int(args.lasso_inner_cv),
                lasso_cv_select=str(args.lasso_cv_select),
                lasso_min_nonzero=int(args.lasso_min_nonzero),
                seed=int(args.seed),
                outdir=outdir,
            )
            pd.DataFrame({"feature": selected}).to_csv(os.path.join(outdir, "selected_features_all.csv"), index=False)
            metrics = _fit_and_eval_models(
                X_train=X_train_all[selected],
                y_train=y_train,
                X_val=X_val_all[selected],
                y_val=y_val,
                X_test=X_test_all[selected],
                y_test=y_test,
                seed=int(args.seed),
                outdir=outdir,
            )
            meta = {
                "pipeline": "threeway_trainvaltest",
                "modality": str(args.modality),
                "labels": str(args.labels),
                "train_ids_path": str(args.train_ids_path),
                "val_ids_path": str(args.val_ids_path),
                "test_ids_path": str(args.test_ids_path),
                "n_train": int(idx_train.size),
                "n_val": int(idx_val.size),
                "n_test": int(idx_test.size),
                "ibsi_exclude_json": str(args.ibsi_exclude_json),
                "ibsi_exclude_sha256": ibsi_sha,
                "feature_selection": sel_meta,
                "models": metrics,
            }

        with open(os.path.join(outdir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return

    splits = _split_indices(
        case_ids=case_ids,
        y=y,
        folds=int(args.folds),
        seed=int(args.seed),
        use_fold_assignments=bool(args.use_fold_assignments),
        fold_assignments_path=str(args.fold_assignments_path),
    )
    if not splits:
        raise ValueError("No valid folds generated")

    lasso_cfg = LassoConfig(
        cs=_parse_float_list(str(args.lasso_cs)),
        min_features=int(args.lasso_min_features),
        max_features=int(args.lasso_max_features),
        target_features=int(args.lasso_target_features),
        max_iter=int(args.lasso_max_iter),
    )
    lasso_penalty = str(args.lasso_penalty).strip().lower()
    lasso_l1_ratios = _parse_float_list(str(args.lasso_l1_ratios))

    oof_rows: list[dict[str, Any]] = []
    stable_rows: list[str] = []
    selection_log: list[dict[str, Any]] = []

    ibsi_rules = _load_ibsi_exclude_rules(str(args.ibsi_exclude_json)) if str(args.pipeline) != "legacy" else []
    extra_regexes = [r.strip() for r in str(args.ibsi_extra_regex).split(",") if r.strip()] if str(args.pipeline) != "legacy" else []
    ibsi_sha = _sha256_file(str(args.ibsi_exclude_json)) if str(args.pipeline) != "legacy" else None

    if str(args.pipeline) != "legacy":
        # Global IBSI Filter (Task B)
        X_df, _, kept_s1, removed_df = _apply_ibsi_filter_by_train(
            X_df,
            pd.DataFrame(),
            extra_regexes=extra_regexes,
            ibsi_rules=ibsi_rules,
        )
        removed_df.to_csv(os.path.join(outdir, "step1_ibsi_removed_features.csv"), index=False)
        pd.DataFrame({"feature": kept_s1}).to_csv(
            os.path.join(outdir, "step1_ibsi_kept_features.txt"), index=False, header=False
        )

    for fold_id, idx_train, idx_test in splits:
        X_train = X_df.iloc[idx_train].copy()
        X_test = X_df.iloc[idx_test].copy()

        if str(args.pipeline) == "legacy":
            X_train_f, X_test_f, kept_cols, meta_filter = _filter_columns_by_train_stats(
                X_train,
                X_test,
                missing_rate_max=float(args.missing_rate_max),
                variance_min=float(args.variance_min),
                corr_method=str(args.corr_method),
                corr_threshold=float(args.corr_threshold),
            )

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_tr_imp = imputer.fit_transform(X_train_f)
            X_tr_sc = scaler.fit_transform(X_tr_imp)
            X_te_sc = scaler.transform(imputer.transform(X_test_f))

            _, sel_names, sel_abscoef, sel_meta, clf = _select_by_l1_logistic(
                X_tr=X_tr_sc,
                y_tr=y[idx_train],
                feature_names=kept_cols,
                cfg=lasso_cfg,
                seed=int(args.seed) + int(fold_id) * 101,
            )
            stable_rows.extend(sel_names)
            selection_log.append(
                {
                    "fold": int(fold_id),
                    **meta_filter,
                    **sel_meta,
                    "n_final_selected": int(len(sel_names)),
                }
            )

            pd.DataFrame({"feature": sel_names, "abs_coef": sel_abscoef}).to_csv(
                os.path.join(outdir, f"selected_features_fold{int(fold_id)}.csv"), index=False
            )

            score = decision_function_pos(clf, X_te_sc).astype(float)
            proba = predict_proba_pos(clf, X_te_sc).astype(float)
            for i, row_idx in enumerate(idx_test.tolist()):
                oof_rows.append(
                    {
                        str(args.id_col): str(case_ids[row_idx]),
                        str(args.label_col): int(y[row_idx]),
                        "fold": int(fold_id),
                        "omics_score": float(score[i]),
                        "omics_proba": float(proba[i]),
                    }
                )
        else:
            fold_dir = os.path.join(outdir, f"fold{int(fold_id)}")
            os.makedirs(fold_dir, exist_ok=True)

            # Task B: IBSI filter is now global (before loop).
            # X_train is already filtered.
            X_train_s1 = X_train
            kept_s1 = X_train.columns.tolist()

            kept_s2, stats_df, meta_ut = _u_test_prefilter_train_only(
                X_train_s1,
                y[idx_train],
                p_threshold=float(args.u_test_p_threshold),
                compute_fdr=bool(args.u_test_fdr),
            )
            stats_df.to_csv(os.path.join(fold_dir, "step2_u_test_stats.csv"), index=False)
            pd.DataFrame({"feature": kept_s2}).to_csv(
                os.path.join(fold_dir, "step2_u_test_kept_features.txt"), index=False, header=False
            )

            X_train_s2 = X_train_s1[kept_s2].copy()

            cv_df = _auc_cv_over_grid(
                X_train_s2,
                y[idx_train],
                penalty=lasso_penalty,
                cs=lasso_cfg.cs,
                l1_ratios=lasso_l1_ratios,
                folds=int(args.lasso_inner_cv),
                seed=int(args.seed) + int(fold_id) * 101,
                max_iter=int(lasso_cfg.max_iter),
            )
            cv_df.to_csv(os.path.join(fold_dir, "step3_lasso_cv_curve.csv"), index=False)
            if str(args.lasso_cv_select) == "best_auc":
                row = cv_df.sort_values(["mean_auc", "C", "l1_ratio"], ascending=[False, True, True]).iloc[0]
                chosen_C = float(row["C"])
                chosen_l1_ratio = float(row.get("l1_ratio", 1.0))
            else:
                chosen_C, chosen_l1_ratio = _choose_grid_by_1se(cv_df)

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_train_s2))
            if lasso_penalty == "elasticnet":
                clf = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=float(chosen_l1_ratio),
                    max_iter=int(lasso_cfg.max_iter),
                    class_weight="balanced",
                    C=float(chosen_C),
                    random_state=int(args.seed) + int(fold_id) * 101,
                )
            else:
                clf = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=int(lasso_cfg.max_iter),
                    class_weight="balanced",
                    C=float(chosen_C),
                    random_state=int(args.seed) + int(fold_id) * 101,
                )
            clf.fit(X_tr_sc, y[idx_train])
            coef = np.ravel(clf.coef_).astype(np.float64)
            abs_coef = np.abs(coef)
            nz = np.where(abs_coef > 1e-12)[0]
            if nz.size == 0:
                chosen_idx = np.argsort(-abs_coef)[: max(int(args.lasso_min_nonzero), 1)]
            else:
                chosen_idx = nz
                if chosen_idx.size < int(args.lasso_min_nonzero):
                    chosen_idx = np.argsort(-abs_coef)[: int(args.lasso_min_nonzero)]

            chosen_idx = np.array(sorted(set(chosen_idx.tolist())), dtype=int)
            sel_names = [kept_s2[i] for i in chosen_idx.tolist()]
            sel_abscoef = abs_coef[chosen_idx].tolist()

            pd.DataFrame({"feature": sel_names}).to_csv(
                os.path.join(fold_dir, "step3_lasso_selected_features.txt"), index=False, header=False
            )
            pd.DataFrame({"feature": sel_names, "abs_coef": sel_abscoef}).to_csv(
                os.path.join(fold_dir, "step3_lasso_coefficients.csv"), index=False
            )

            stable_rows.extend(sel_names)
            selection_log.append(
                {
                    "fold": int(fold_id),
                    "step1_kept": int(len(kept_s1)),
                    "step1_removed": int(removed_df["feature"].nunique()) if not removed_df.empty else 0,
                    **meta_ut,
                    "lasso_penalty": str(lasso_penalty),
                    "lasso_l1_ratio": float(chosen_l1_ratio),
                    "lasso_C": float(chosen_C),
                    "lasso_cv_select": str(args.lasso_cv_select),
                    "n_final_selected": int(len(sel_names)),
                }
            )

    freq_df = pd.Series(stable_rows).value_counts().reset_index()
    freq_df.columns = ["feature", "count"]
    freq_df["freq"] = freq_df["count"] / float(len(splits))
    freq_df = freq_df.sort_values(["count", "feature"], ascending=[False, True]).reset_index(drop=True)
    freq_df.to_csv(os.path.join(outdir, "stable_features_frequency.csv"), index=False)

    min_folds = int(args.stable_min_folds)
    stable_df = freq_df[freq_df["count"] >= min_folds][["feature"]].copy()
    stable_path = os.path.join(outdir, f"stable_features_min{min_folds}of{int(args.folds)}.csv")
    stable_df.to_csv(stable_path, index=False)

    drift_gate_enabled = bool(str(args.drift_diagnostics_path).strip()) and float(args.drift_ks_max) > 0.0
    drift_removed = 0
    drift_removed_in_top = 0
    drift_split_key = str(args.drift_split_key)
    drift_ks_max = float(args.drift_ks_max)
    if drift_gate_enabled:
        drift_top = _load_drift_top_ks(path=str(args.drift_diagnostics_path), split_key=drift_split_key)
        stable_feats = stable_df["feature"].astype(str).tolist()
        keep_feats: list[str] = []
        removed_feats: list[str] = []
        for f in stable_feats:
            ks = drift_top.get(str(f))
            if ks is not None and float(ks) > drift_ks_max:
                removed_feats.append(str(f))
            else:
                keep_feats.append(str(f))
        drift_removed = int(len(removed_feats))
        drift_removed_in_top = int(len([f for f in removed_feats if f in drift_top]))

        stable_df2 = pd.DataFrame({"feature": keep_feats})
        if stable_df2.empty:
            raise ValueError(
                f"漂移门槛过滤后稳定特征为空：split={drift_split_key}, ks_max={drift_ks_max}, "
                f"原始稳定特征数={len(stable_feats)}, 被剔除={drift_removed}"
            )
        ks_tag = _format_float_for_path(drift_ks_max)
        stable_path = os.path.join(outdir, f"stable_features_min{min_folds}of{int(args.folds)}_ks{ks_tag}.csv")
        stable_df2.to_csv(stable_path, index=False)
        stable_df = stable_df2

    if str(args.pipeline) == "legacy" and oof_rows:
        oof_df = pd.DataFrame(oof_rows)
        oof_df = oof_df.sort_values([str(args.id_col)]).reset_index(drop=True)
        oof_df.to_csv(os.path.join(outdir, "omics_scores_oof.csv"), index=False)

    kept_final = [c for c in X_df.columns.tolist() if c in set(stable_df["feature"].astype(str).tolist())]
    if kept_final and str(args.pipeline) == "legacy":
        imputer_g = SimpleImputer(strategy="median")
        scaler_g = StandardScaler()
        X_imp_g = imputer_g.fit_transform(X_df[kept_final])
        X_sc_g = scaler_g.fit_transform(X_imp_g)
        _, sel_names_g, sel_abscoef_g, sel_meta_g, clf_g = _select_by_l1_logistic(
            X_tr=X_sc_g,
            y_tr=y,
            feature_names=kept_final,
            cfg=lasso_cfg,
            seed=int(args.seed) + 999,
        )
        global_bundle = {
            "modality": str(args.modality),
            "feature_cols": kept_final,
            "selected_features": sel_names_g,
            "lasso_meta": sel_meta_g,
            "imputer": imputer_g,
            "scaler": scaler_g,
            "model": clf_g,
        }
        joblib.dump(global_bundle, os.path.join(outdir, "lasso_global_bundle.joblib"))

        pd.DataFrame({"feature": sel_names_g, "abs_coef": sel_abscoef_g}).to_csv(
            os.path.join(outdir, "selected_features_global.csv"), index=False
        )

        g_score = decision_function_pos(clf_g, X_sc_g).astype(float)
        g_proba = predict_proba_pos(clf_g, X_sc_g).astype(float)
        pd.DataFrame(
            {
                str(args.id_col): case_ids.tolist(),
                str(args.label_col): y.tolist(),
                "omics_score": g_score.tolist(),
                "omics_proba": g_proba.tolist(),
            }
        ).to_csv(os.path.join(outdir, "omics_scores_globalfit.csv"), index=False)

    meta = {
        "pipeline": str(args.pipeline),
        "modality": str(args.modality),
        "inputs": input_paths,
        "labels": str(args.labels),
        "id_col": str(args.id_col),
        "label_col_in_labels": str(args.label_col_in_labels),
        "label_col": str(args.label_col),
        "n_samples": int(len(case_ids)),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "use_fold_assignments": bool(args.use_fold_assignments),
        "fold_assignments_path": str(args.fold_assignments_path),
        "ibsi_exclude_json": str(args.ibsi_exclude_json),
        "ibsi_exclude_sha256": ibsi_sha,
        "ibsi_extra_regex": str(args.ibsi_extra_regex),
        "u_test_p_threshold": float(args.u_test_p_threshold),
        "u_test_fdr": bool(args.u_test_fdr),
        "missing_rate_max": float(args.missing_rate_max),
        "variance_min": float(args.variance_min),
        "corr_method": str(args.corr_method),
        "corr_threshold": float(args.corr_threshold),
        "lasso": {
            "cs": lasso_cfg.cs,
            "min_features": int(lasso_cfg.min_features),
            "max_features": int(lasso_cfg.max_features),
            "target_features": int(lasso_cfg.target_features),
            "max_iter": int(lasso_cfg.max_iter),
            "penalty": str(lasso_penalty),
            "l1_ratios": lasso_l1_ratios,
            "inner_cv": int(args.lasso_inner_cv),
            "cv_select": str(args.lasso_cv_select),
            "min_nonzero": int(args.lasso_min_nonzero),
        },
        "stable_min_folds": int(args.stable_min_folds),
        "drift_gate": {
            "enabled": bool(drift_gate_enabled),
            "diagnostics_path": str(args.drift_diagnostics_path),
            "split_key": drift_split_key,
            "ks_max": drift_ks_max if drift_gate_enabled else None,
            "removed": int(drift_removed) if drift_gate_enabled else 0,
            "removed_in_diagnostics_top": int(drift_removed_in_top) if drift_gate_enabled else 0,
        },
        "selection_log": selection_log,
        "stable_features_path": stable_path,
        "global_fit": {
            "enabled": bool(kept_final) and str(args.pipeline) == "legacy",
            "n_stable_features_in_matrix": int(len(kept_final)),
        },
    }
    if prev_run_meta is not None:
        changes: list[dict[str, Any]] = []
        pairs: list[tuple[str, Any, Any]] = [
            ("u_test_p_threshold", prev_run_meta.get("u_test_p_threshold"), meta.get("u_test_p_threshold")),
            ("corr_threshold", prev_run_meta.get("corr_threshold"), meta.get("corr_threshold")),
            ("missing_rate_max", prev_run_meta.get("missing_rate_max"), meta.get("missing_rate_max")),
            ("variance_min", prev_run_meta.get("variance_min"), meta.get("variance_min")),
            (
                "stable_min_folds",
                prev_run_meta.get("stable_min_folds"),
                meta.get("stable_min_folds"),
            ),
            (
                "lasso.target_features",
                (prev_run_meta.get("lasso") or {}).get("target_features"),
                (meta.get("lasso") or {}).get("target_features"),
            ),
            (
                "lasso.max_features",
                (prev_run_meta.get("lasso") or {}).get("max_features"),
                (meta.get("lasso") or {}).get("max_features"),
            ),
            (
                "lasso.min_features",
                (prev_run_meta.get("lasso") or {}).get("min_features"),
                (meta.get("lasso") or {}).get("min_features"),
            ),
        ]
        for k, old, new in pairs:
            if old != new:
                changes.append({"param": k, "from": old, "to": new})
        if changes:
            meta["relaxation"] = {
                "previous_run_meta_path": prev_run_meta_path,
                "changes": changes,
            }
    with open(os.path.join(outdir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

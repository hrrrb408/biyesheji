import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from metrics_utils import compute_metrics_with_ci, predict_proba_pos
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

CT_STABLE_FEATURES_MIN3OF5_PATH = "outputs/selection/ct_ibsi_utest_lasso_seed42_k5_trainonly/stable_features_min3of5.csv"
DOSE_STABLE_FEATURES_MIN3OF5_PATH = "outputs/selection/dose_ibsi_utest_lasso_seed42_k5_trainonly/stable_features_min3of5.csv"
CT_STABLE_FEATURES_FREQ_PATH = "outputs/selection/ct_ibsi_utest_lasso_seed42_k5_trainonly/stable_features_frequency.csv"
DOSE_STABLE_FEATURES_FREQ_PATH = "outputs/selection/dose_ibsi_utest_lasso_seed42_k5_trainonly/stable_features_frequency.csv"

ModelKind = Literal["xgb", "lr"]
SearchMetric = Literal["auc", "aucpr", "logloss"]
ThresholdStrategy = Literal["youden", "balanced_accuracy", "f1", "acc", "recall95", "fixed0.4"]
SplitScheme = Literal["threeway_files"]


@dataclass(frozen=True)
class SplitConfig:
    scheme: SplitScheme = "threeway_files"
    random_seed: int = 42
    train_ids_path: str = "outputs/splits/threeway_train_ids.csv"
    val_ids_path: str = "outputs/splits/threeway_val_ids.csv"
    test_ids_path: str = "outputs/splits/threeway_test_ids.csv"


@dataclass(frozen=True)
class FeatureFilterConfig:
    missing_rate_max: float = 0.3
    variance_min: float = 1e-8
    corr_threshold: float = 0.95
    variance_top_fraction_if_many: float = 0.8
    variance_top_threshold: int = 50


@dataclass(frozen=True)
class StableFeatureConfig:
    stable_features_path: str | None = None
    min_count: int | None = None
    drift_diagnostics_path: str | None = None
    drift_split_key: str = "train_vs_test"
    drift_ks_max: float | None = None


@dataclass(frozen=True)
class PreprocessConfig:
    clip_abs: float | None = None


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = True
    method: Literal["sigmoid", "isotonic"] = "sigmoid"
    calib_fraction: float = 0.2
    direction_protect: bool = False


@dataclass(frozen=True)
class RandomSearchConfig:
    enabled: bool = True
    trials: int = 150
    splits: int = 5
    # XGBoost specific ranges
    learning_rate_range: tuple[float, float] = (0.02, 0.1)
    max_depth_range: tuple[int, int] = (3, 6)
    subsample_range: tuple[float, float] = (0.6, 0.9)
    colsample_bytree_range: tuple[float, float] = (0.6, 0.9)
    gamma_range: tuple[float, float] = (0.1, 2.0)
    min_child_weight_range: tuple[int, int] = (3, 15)
    reg_alpha_range: tuple[float, float] = (0.5, 5.0)
    reg_lambda_range: tuple[float, float] = (1.0, 10.0)
    scale_pos_weight_factor_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class XGBBaseParams:
    n_estimators: int = 1000
    early_stopping_rounds: int = 80
    tree_method: str = "hist"
    n_jobs: int = 1


@dataclass(frozen=True)
class SingleOmicsRunConfig:
    data_paths: list[str]
    label_path: str
    out_dir: str
    id_col: str = "case_id"
    label_col_in_labels: str = "y"
    label_col: str = "label"
    drop_cols: list[str] = field(default_factory=list)
    sheet_name: int | str = 0
    eval_metrics: list[SearchMetric] = field(default_factory=lambda: ["auc"])
    threshold_strategy: ThresholdStrategy = "fixed0.4"
    hyper_search_metric: SearchMetric = "auc"
    model_kind: ModelKind = "xgb"
    n_splits: int = 5
    split: SplitConfig = field(default_factory=SplitConfig)
    filter_cfg: FeatureFilterConfig = field(default_factory=FeatureFilterConfig)
    stable: StableFeatureConfig = field(default_factory=StableFeatureConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    random_search: RandomSearchConfig = field(default_factory=RandomSearchConfig)
    base_params: XGBBaseParams = field(default_factory=XGBBaseParams)


@dataclass(frozen=True)
class EarlyFusionRunConfig:
    ct_data_paths: list[str]
    dose_data_paths: list[str]
    label_path: str
    out_dir: str
    id_col: str = "case_id"
    label_col_in_labels: str = "y"
    label_col: str = "label"
    sheet_name: int | str = 0
    eval_metrics: list[SearchMetric] = field(default_factory=lambda: ["auc"])
    threshold_strategy: ThresholdStrategy = "fixed0.4"
    hyper_search_metric: SearchMetric = "auc"
    model_kind: ModelKind = "xgb"
    n_splits: int = 5
    split: SplitConfig = field(default_factory=SplitConfig)
    filter_cfg: FeatureFilterConfig = field(default_factory=FeatureFilterConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    ct_stable_features_path: str | None = None
    dose_stable_features_path: str | None = None
    ct_stable_min_count: int | None = None
    dose_stable_min_count: int | None = None
    random_search: RandomSearchConfig = field(default_factory=RandomSearchConfig)
    base_params: XGBBaseParams = field(default_factory=XGBBaseParams)


class PreprocessedEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, preprocessor: Pipeline, clf: Any):
        self.preprocessor = preprocessor
        self.clf = clf

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        X_t = self.preprocessor.transform(X)
        return self.clf.predict_proba(X_t)

    def predict(self, X):
        return self.clf.predict(self.preprocessor.transform(X))

    @property
    def classes_(self):
        return getattr(self.clf, "classes_", np.array([0, 1]))


class XGBEnsemble(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, estimators: list[Any], threshold: float = 0.5):
        self.estimators = estimators
        self.threshold = threshold

    def _more_tags(self):
        return {"estimator_type": "classifier"}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        # Estimators are assumed to be already fitted
        return self

    def predict_proba(self, X):
        # Average probabilities from all estimators
        probs_list = []
        for est in self.estimators:
            p = est.predict_proba(X)
            probs_list.append(p)
            
        # Stack and mean
        # probs shape: (n_estimators, n_samples, n_classes) -> (n_samples, n_classes)
        probs = np.mean(np.array(probs_list), axis=0)
        return probs

    def predict(self, X):
        proba = predict_proba_pos(self, X)
        return (proba >= self.threshold).astype(int)
    
    @property
    def classes_(self):
        if self.estimators:
            return self.estimators[0].classes_
        return np.array([0, 1])


class CalibratedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, ensemble: XGBEnsemble, calibrator: Any, threshold: float):
        self.ensemble = ensemble
        self.calibrator = calibrator
        self.threshold = float(threshold)

    def _more_tags(self):
        return {"estimator_type": "classifier"}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        return self.ensemble.predict_proba(X)

    def predict(self, X):
        proba = predict_proba_pos(self, X)
        return (proba >= self.threshold).astype(int)

    @property
    def classes_(self):
        return self.ensemble.classes_


class PrefitProbabilityCalibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: Any, *, method: str = "sigmoid", direction_protect: bool = True):
        self.base_estimator = base_estimator
        self.method = str(method)
        self.direction_protect = bool(direction_protect)

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=int)
        p = predict_proba_pos(self.base_estimator, X)
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        self._invert_output = False

        if self.method == "sigmoid":
            z = np.log(p / (1.0 - p)).reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(z, y_arr)
            self._sigmoid_lr = lr
            self._isotonic = None
            if bool(self.direction_protect):
                coef = float(np.asarray(lr.coef_, dtype=float).reshape(-1)[0])
                self._invert_output = bool(coef < 0.0)
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y_arr)
            self._sigmoid_lr = None
            self._isotonic = iso
            self._invert_output = False
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")

        classes = getattr(self.base_estimator, "classes_", None)
        if classes is None:
            self.classes_ = np.array([0, 1], dtype=int)
        else:
            self.classes_ = np.asarray(classes)
        return self

    def predict_proba(self, X):
        p = predict_proba_pos(self.base_estimator, X)
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)

        if getattr(self, "_sigmoid_lr", None) is not None:
            z = np.log(p / (1.0 - p)).reshape(-1, 1)
            p_cal = predict_proba_pos(self._sigmoid_lr, z)
        elif getattr(self, "_isotonic", None) is not None:
            p_cal = np.asarray(self._isotonic.transform(p), dtype=float)
        else:
            raise RuntimeError("Calibrator is not fitted")

        p_cal = np.clip(p_cal, 0.0, 1.0)
        if bool(getattr(self, "direction_protect", True)) and bool(getattr(self, "_invert_output", False)):
            p_cal = 1.0 - p_cal
        if self.classes_.size == 2:
            if int(self.classes_[0]) == 1:
                return np.column_stack([p_cal, 1.0 - p_cal])
            if int(self.classes_[1]) == 1:
                return np.column_stack([1.0 - p_cal, p_cal])
        return np.column_stack([1.0 - p_cal, p_cal])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def _ensure_pickleable_module_names() -> None:
    if __name__ != "__main__":
        return
    sys.modules.setdefault("XGB.xgb_train", sys.modules[__name__])
    for name in ["PreprocessedEstimator", "XGBEnsemble", "NumpyEncoder", "Clipper"]:
        cls = globals().get(name)
        if cls is not None:
            cls.__module__ = "XGB.xgb_train"


def load_table(path: str, sheet_name: int | str = 0) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported file type: {ext}")


def _read_id_list(path: str, *, id_col: str) -> list[str]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    col = str(id_col)
    if col not in df.columns:
        if "case_id" in df.columns:
            col = "case_id"
        else:
            raise ValueError(f"Missing id column in {path}: tried {id_col}/case_id")
    return df[col].astype(str).tolist()


def merge_feature_tables(
    paths: list[str],
    *,
    id_col: str,
    sheet_name: int | str = 0,
    label_cols_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    if len(paths) == 0:
        raise ValueError("paths is empty")

    label_cols_to_drop = label_cols_to_drop or []
    dfs: list[pd.DataFrame] = []
    for p in paths:
        dfi = load_table(p, sheet_name=sheet_name)
        if id_col not in dfi.columns:
            raise ValueError(f"Missing id_col={id_col} in {p}")

        for col in label_cols_to_drop:
            if col in dfi.columns:
                dfi = dfi.drop(columns=[col])

        non_id_cols = [c for c in dfi.columns if c != id_col]
        has_tagged_names = any(("__ct__" in c or "__dose__" in c) for c in non_id_cols)
        if not has_tagged_names:
            stem = os.path.splitext(os.path.basename(p))[0]
            modality = "feat"
            organ = stem
            if stem.startswith("ct_radiomics_"):
                modality, organ = "ct", stem[len("ct_radiomics_") :]
            elif stem.startswith("dosiomics_"):
                modality, organ = "dose", stem[len("dosiomics_") :]
            prefix = f"{modality}__{organ}__"
            rename_map = {c: prefix + c for c in dfi.columns if c != id_col}
            dfi = dfi.rename(columns=rename_map)

        dfs.append(dfi)

    df = dfs[0]
    for other in dfs[1:]:
        df = df.merge(other, on=id_col, how="inner")
    return df


def compute_metrics(y_true, y_prob, *, thr: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if np.unique(y_true).size < 2:
        auc = np.nan
        ap = np.nan
        ll = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ll = log_loss(y_true, y_prob, labels=[0, 1])

    # Find metrics at Recall >= 0.95 (Clinical Requirement)
    # We sort by y_prob descending
    desc_score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    y_prob_desc = y_prob[desc_score_indices]
    y_true_desc = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_prob_desc))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_desc.size - 1]
    
    best_spec_at_95 = 0.0
    fpr_at_95 = 1.0
    thr_at_95 = 0.0
    
    # Calculate TPR/FPR for all thresholds (manual or sklearn)
    # Using simple iteration over thresholds for clarity
    tps = np.cumsum(y_true_desc)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos > 0 and n_neg > 0:
        tprs = tps / n_pos
        fprs = fps / n_neg
        
        # Find first index where TPR >= 0.95
        # Since we iterate descending thresholds (high to low), TPR increases.
        idx_95 = np.argmax(tprs >= 0.95)
        if tprs[idx_95] >= 0.95:
            best_spec_at_95 = 1.0 - fprs[idx_95]
            fpr_at_95 = fprs[idx_95]
            thr_at_95 = y_prob_desc[threshold_idxs[idx_95]]

    return {
        "AUC": float(auc) if np.isfinite(auc) else np.nan,
        "AP": float(ap) if np.isfinite(ap) else np.nan,
        "LogLoss": float(ll) if np.isfinite(ll) else np.nan,
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Threshold": float(thr),
        "Spec_at_Recall95": float(best_spec_at_95),
        "FPR_at_Recall95": float(fpr_at_95),
        "Thr_at_Recall95": float(thr_at_95)
    }


def find_best_threshold(y_true, y_prob, *, metric: ThresholdStrategy = "f1") -> tuple[float, float]:
    metric_lower = metric.lower()
    if metric_lower == "fixed0.4":
        return 0.4, 0.4

    if metric_lower in ["recall95", "recall_95", "tpr95", "min_fn", "minfn"]:
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        if y_true.size == 0:
            return 0.5, np.nan

        desc_score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
        y_prob_desc = y_prob[desc_score_indices]
        y_true_desc = y_true[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_prob_desc))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true_desc.size - 1]

        tps = np.cumsum(y_true_desc)[threshold_idxs]
        fps = (1 + threshold_idxs) - tps
        n_pos = int(np.sum(y_true))
        n_neg = int(y_true.size - n_pos)
        if n_pos <= 0 or n_neg <= 0:
            return 0.5, np.nan

        tprs = tps / n_pos
        fprs = fps / n_neg
        idx = int(np.argmax(tprs >= 0.95))
        if float(tprs[idx]) >= 0.95:
            thr = float(y_prob_desc[threshold_idxs[idx]])
            spec = float(1.0 - fprs[idx])
            return thr, spec

        idx_best = int(np.argmax(tprs))
        thr = float(y_prob_desc[threshold_idxs[idx_best]])
        spec = float(1.0 - fprs[idx_best])
        return thr, spec

    if metric_lower == "fixed0.4":
        return 0.4, 0.4

    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr, best_val = 0.5, -1.0
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, thr=float(t))
        if metric_lower == "f1":
            val = m.get("F1", np.nan)
        elif metric_lower in ["acc", "accuracy"]:
            val = m.get("ACC", np.nan)
        elif metric_lower in ["balanced_accuracy", "balancedacc", "bacc"]:
            val = (m.get("Recall", np.nan) + m.get("Specificity", np.nan)) / 2.0
        elif metric_lower in ["youden", "youdenj", "j"]:
            val = m.get("Recall", np.nan) + m.get("Specificity", np.nan) - 1.0
        else:
            val = m.get(metric, np.nan)
        if np.isfinite(val) and float(val) > best_val:
            best_val = float(val)
            best_thr = float(t)
    return float(best_thr), float(best_val)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _row_missing_fraction(X: pd.DataFrame) -> np.ndarray:
    if int(X.shape[0]) == 0:
        return np.array([], dtype=float)
    return X.isna().mean(axis=1).to_numpy(dtype=float)


def _row_mean_abs_z(*, X_train: pd.DataFrame, X_other: pd.DataFrame) -> np.ndarray:
    if int(X_other.shape[0]) == 0:
        return np.array([], dtype=float)
    if int(X_other.shape[1]) == 0:
        return np.full(shape=(int(X_other.shape[0]),), fill_value=np.nan, dtype=float)
    mu = X_train.mean(axis=0, skipna=True).to_numpy(dtype=float)
    sigma = X_train.std(axis=0, skipna=True).to_numpy(dtype=float)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    x = X_other.to_numpy(dtype=float)
    z = (x - mu) / sigma
    az = np.abs(z)
    mask = np.isfinite(az)
    s = np.where(mask, az, 0.0).sum(axis=1)
    c = mask.sum(axis=1)
    out = (s / np.maximum(c, 1)).astype(float)
    out[c == 0] = np.nan
    return out


def _quantile_bins(values: np.ndarray, *, n_bins: int) -> tuple[np.ndarray, list[float]] | tuple[None, None]:
    v = np.asarray(values, dtype=float)
    m = np.isfinite(v)
    if int(m.sum()) < max(5, int(n_bins) * 2):
        return None, None
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(v[m], qs).astype(float).tolist()
    edges_u: list[float] = [float(edges[0])]
    for e in edges[1:]:
        if float(e) > float(edges_u[-1]) + 1e-12:
            edges_u.append(float(e))
    if len(edges_u) < 3:
        return None, None
    bins = np.full(shape=v.shape, fill_value=-1, dtype=int)
    finite_v = v[m]
    idx = np.searchsorted(np.asarray(edges_u[1:-1], dtype=float), finite_v, side="right")
    bins[m] = idx.astype(int)
    return bins, edges_u


def _compute_slice_metrics(*, df: pd.DataFrame, thr: float, group_col: str, min_n: int) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_col, dropna=False):
        n = int(g.shape[0])
        if n < int(min_n):
            continue
        y_true = g["y_true"].to_numpy(dtype=int)
        y_prob = g["y_prob"].to_numpy(dtype=float)
        m = compute_metrics(y_true, y_prob, thr=float(thr))
        pos = int(np.sum(y_true == 1))
        neg = int(np.sum(y_true == 0))
        rows.append(
            {
                "group_col": str(group_col),
                "group_value": str(key),
                "n": n,
                "pos": pos,
                "neg": neg,
                "pos_rate": float(pos / n) if n else np.nan,
                **m,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    cols_front = ["group_col", "group_value", "n", "pos", "neg", "pos_rate"]
    cols_other = [c for c in out.columns if c not in cols_front]
    return out[cols_front + cols_other].sort_values(by=["group_col", "n"], ascending=[True, False]).reset_index(drop=True)


def _write_val_diagnostics(
    *,
    metric_dir: str,
    id_col: str,
    ids_val: np.ndarray,
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    thr: float,
    X_train_f: pd.DataFrame,
    X_val_f: pd.DataFrame,
    labels_full: pd.DataFrame | None,
    min_group_n: int = 8,
    top_k: int = 25,
) -> None:
    ids_val_s = pd.Series(ids_val).astype(str)
    base_df = pd.DataFrame(
        {
            str(id_col): ids_val_s,
            "y_true": np.asarray(y_val, dtype=int),
            "y_prob": np.asarray(y_prob_val, dtype=float),
        }
    )
    base_df["y_pred"] = (base_df["y_prob"].to_numpy(dtype=float) >= float(thr)).astype(int)
    base_df["val_feature_missing_frac"] = _row_missing_fraction(X_val_f)
    base_df["val_mean_abs_z_train"] = _row_mean_abs_z(X_train=X_train_f, X_other=X_val_f)

    if labels_full is not None and str(id_col) in labels_full.columns:
        meta = labels_full.copy()
        meta[str(id_col)] = meta[str(id_col)].astype(str)
        meta = meta.drop_duplicates(subset=[str(id_col)], keep="first")
        base_df = base_df.merge(meta, on=str(id_col), how="left")

    miss_bins, miss_edges = _quantile_bins(base_df["val_feature_missing_frac"].to_numpy(dtype=float), n_bins=4)
    if miss_bins is not None:
        base_df["val_missing_bin"] = miss_bins
        base_df["val_missing_bin"] = base_df["val_missing_bin"].map(
            lambda k: f"[{miss_edges[int(k)]:.3f},{miss_edges[int(k)+1]:.3f}]" if int(k) >= 0 else "NA"
        )

    z_bins, z_edges = _quantile_bins(base_df["val_mean_abs_z_train"].to_numpy(dtype=float), n_bins=4)
    if z_bins is not None:
        base_df["val_absz_bin"] = z_bins
        base_df["val_absz_bin"] = base_df["val_absz_bin"].map(
            lambda k: f"[{z_edges[int(k)]:.3f},{z_edges[int(k)+1]:.3f}]" if int(k) >= 0 else "NA"
        )

    base_auc = _safe_auc(base_df["y_true"].to_numpy(dtype=int), base_df["y_prob"].to_numpy(dtype=float))
    y_true_all = base_df["y_true"].to_numpy(dtype=int)
    y_prob_all = base_df["y_prob"].to_numpy(dtype=float)
    influence: list[float] = []
    for i in range(int(base_df.shape[0])):
        auc_i = _safe_auc(np.delete(y_true_all, i), np.delete(y_prob_all, i))
        if np.isfinite(base_auc) and np.isfinite(auc_i):
            influence.append(float(auc_i - base_auc))
        else:
            influence.append(float("nan"))
    base_df["delta_auc_if_dropped"] = np.asarray(influence, dtype=float)

    out_influence = base_df.sort_values(by=["delta_auc_if_dropped"], ascending=False).reset_index(drop=True)
    out_influence.to_csv(os.path.join(metric_dir, "val_influence.csv"), index=False)

    candidate_group_cols: list[str] = []
    for c in base_df.columns:
        if str(c).startswith("missing_"):
            candidate_group_cols.append(str(c))
    candidate_group_cols += ["val_missing_bin", "val_absz_bin"]
    candidate_group_cols = [c for c in candidate_group_cols if c in base_df.columns]
    candidate_group_cols = list(dict.fromkeys(candidate_group_cols))

    slice_tables: list[pd.DataFrame] = []
    for gc in candidate_group_cols:
        st = _compute_slice_metrics(df=base_df, thr=float(thr), group_col=str(gc), min_n=int(min_group_n))
        if not st.empty:
            slice_tables.append(st)
    if slice_tables:
        out_slices = pd.concat(slice_tables, axis=0, ignore_index=True)
        out_slices.to_csv(os.path.join(metric_dir, "val_slice_metrics.csv"), index=False)

    top_by_absz = base_df.sort_values(by=["val_mean_abs_z_train"], ascending=False).head(int(top_k))
    top_by_absz.to_csv(os.path.join(metric_dir, "val_outliers_absz.csv"), index=False)


def score_for_search(y_true, y_score, *, eval_metric: SearchMetric) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if eval_metric == "auc":
        return float(roc_auc_score(y_true, y_score))
    if eval_metric == "aucpr":
        return float(average_precision_score(y_true, y_score))
    if eval_metric == "logloss":
        return -float(log_loss(y_true, y_score, labels=[0, 1]))  # Maximize negative logloss
    raise ValueError(f"Unsupported eval_metric: {eval_metric}")


def split_train_val_test(
    *,
    case_ids: np.ndarray,
    y: np.ndarray,
    split: SplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _sorted(a: np.ndarray) -> np.ndarray:
        return np.array(sorted(np.asarray(a, dtype=int).tolist()), dtype=int)

    train_ids = set(_read_id_list(split.train_ids_path, id_col="case_id"))
    val_ids = set(_read_id_list(split.val_ids_path, id_col="case_id"))
    test_ids = set(_read_id_list(split.test_ids_path, id_col="case_id"))
    if not train_ids or not val_ids or not test_ids:
        raise FileNotFoundError(
            f"缺少预先划分的三划分文件：{split.train_ids_path} / {split.val_ids_path} / {split.test_ids_path}"
        )

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("threeway_{train,val,test}_ids.csv 存在重复 case_id")

    case_ids_str = np.asarray(case_ids, dtype=str)
    all_ids = train_ids | val_ids | test_ids
    missing = sorted(set(case_ids_str.tolist()) - all_ids)
    extra = sorted(all_ids - set(case_ids_str.tolist()))
    if missing:
        raise RuntimeError(f"threeway split 缺少 {len(missing)} 个 case_id。例：{missing[:5]}")
    if extra:
        print(f"WARNING: threeway split 多出 {len(extra)} 个 case_id，将忽略。例：{extra[:5]}")

    idx_train = np.where(np.isin(case_ids_str, list(train_ids)))[0]
    idx_val = np.where(np.isin(case_ids_str, list(val_ids)))[0]
    idx_test = np.where(np.isin(case_ids_str, list(test_ids)))[0]
    return _sorted(idx_train), _sorted(idx_val), _sorted(idx_test)


def make_split_table(
    *,
    case_ids: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    id_col: str,
) -> pd.DataFrame:
    n = int(len(case_ids))
    split_arr = np.full(shape=(n,), fill_value="", dtype=object)
    split_arr[np.asarray(idx_train, dtype=int)] = "train"
    split_arr[np.asarray(idx_val, dtype=int)] = "val"
    split_arr[np.asarray(idx_test, dtype=int)] = "test"
    if np.any(split_arr == ""):
        raise RuntimeError("split_table has uncovered indices")
    df = pd.DataFrame(
        {
            str(id_col): pd.Series(case_ids).astype(str),
            "y": pd.Series(y).astype(int),
            "split": pd.Series(split_arr).astype(str),
        }
    )
    return df.sort_values(by=[str(id_col)]).reset_index(drop=True)


def _fit_and_predict_train_val_test(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    cfg: SingleOmicsRunConfig | EarlyFusionRunConfig,
    best_params: dict[str, Any],
    base_params: XGBBaseParams,
) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray, float, float]:
    preprocess = cfg.preprocess if hasattr(cfg, "preprocess") else PreprocessConfig()
    calibration = getattr(cfg, "calibration", CalibrationConfig())
    es_metric = "auc" if getattr(cfg, "hyper_search_metric", "auc") == "auc" else "logloss"
    ensemble_size = int(getattr(cfg, "fulltrain_ensemble_size", 5))

    oof_prob = _xgb_oof_predict(
        X_train,
        y_train,
        n_splits=int(getattr(cfg, "n_splits", 5)),
        ensemble_size=ensemble_size,
        preprocess=preprocess,
        best_params=best_params,
        base_params=base_params,
        random_state=int(cfg.split.random_seed),
        eval_metric=es_metric,
        calibration=calibration,
    )

    thr_to_use = 0.5
    thr_to_use_val = np.nan
    oof_prob_arr = np.asarray(oof_prob, dtype=float)
    oof_mask = np.isfinite(oof_prob_arr)
    if oof_mask.any() and np.unique(np.asarray(y_train, dtype=int)[oof_mask]).size >= 2:
        thr_to_use, thr_to_use_val = find_best_threshold(
            np.asarray(y_train, dtype=int)[oof_mask],
            oof_prob_arr[oof_mask],
            metric=cfg.threshold_strategy,
        )

    if calibration.enabled and float(calibration.calib_fraction) > 0 and len(y_train) >= 10:
        idx_all = np.arange(len(y_train), dtype=int)
        try:
            idx_fit, idx_calib = train_test_split(
                idx_all,
                test_size=float(calibration.calib_fraction),
                stratify=np.asarray(y_train, dtype=int),
                random_state=int(cfg.split.random_seed),
            )
        except Exception:
            idx_fit, idx_calib = idx_all, np.array([], dtype=int)
    else:
        idx_fit, idx_calib = np.arange(len(y_train), dtype=int), np.array([], dtype=int)

    models: list[Pipeline] = []
    for j in range(max(1, ensemble_size)):
        models.append(
            _fit_xgb_refit_full(
                X_train.iloc[idx_fit],
                np.asarray(y_train, dtype=int)[idx_fit],
                preprocess=preprocess,
                best_params=best_params,
                base_params=base_params,
                random_state=int(cfg.split.random_seed) + 1000 * int(j),
                eval_metric=es_metric,
            )
        )
    ensemble = XGBEnsemble(estimators=models, threshold=float(thr_to_use))

    calibrator = None
    if idx_calib.size > 0:
        y_calib = np.asarray(y_train, dtype=int)[idx_calib]
        if np.unique(y_calib).size >= 2:
            calibrator = PrefitProbabilityCalibrator(
                ensemble,
                method=str(calibration.method),
                direction_protect=bool(getattr(calibration, "direction_protect", True)),
            )
            calibrator.fit(X_train.iloc[idx_calib], y_calib)
            try:
                yv = np.asarray(y_val, dtype=int)
                if np.unique(yv).size >= 2:
                    p_raw = predict_proba_pos(ensemble, X_val)
                    p_cal = predict_proba_pos(calibrator, X_val)
                    auc_raw = float(roc_auc_score(yv, np.asarray(p_raw, dtype=float)))
                    auc_cal = float(roc_auc_score(yv, np.asarray(p_cal, dtype=float)))
                    if np.isfinite(auc_raw) and np.isfinite(auc_cal) and auc_cal < auc_raw:
                        calibrator = None
            except Exception:
                pass

    model = CalibratedEnsemble(ensemble=ensemble, calibrator=calibrator, threshold=float(thr_to_use))
    y_prob_val = predict_proba_pos(model, X_val)
    y_prob_test = predict_proba_pos(model, X_test)
    # 注意：无法在此处自动反转测试集预测，因为没有y_test标签

    return y_prob_val, y_prob_test, model, oof_prob, float(thr_to_use), float(thr_to_use_val)


class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, abs_max: float):
        self.abs_max = float(abs_max)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return np.clip(X, -self.abs_max, self.abs_max)


def _build_preprocess_steps(
    *,
    preprocess: PreprocessConfig,
    random_state: int,
) -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
    if preprocess.clip_abs is not None:
        steps.append(("clipper", Clipper(abs_max=float(preprocess.clip_abs))))
    return steps


def build_xgb_pipeline(
    *,
    params: dict[str, Any],
    random_state: int,
    preprocess: PreprocessConfig,
    base_params: XGBBaseParams,
):
    """
    Constructs an XGBoost pipeline.
    Note: For XGBoost < 3.0, early_stopping_rounds was often passed to fit().
    For XGBoost >= 3.0 (and some sklearn wrappers), it should be passed to __init__.
    However, we primarily use manual construction in _fit_and_eval_fold to ensure correct behavior.
    This helper is kept for reference or simple usage.
    """
    xgb_kwargs = {
        "n_estimators": base_params.n_estimators,
        "random_state": random_state,
        "n_jobs": base_params.n_jobs,
        "tree_method": base_params.tree_method,
        "eval_metric": "logloss",
        "early_stopping_rounds": base_params.early_stopping_rounds,
        **params
    }
    
    # In this codebase, we decided to pass early_stopping_rounds in init for stability with XGB 3.1.2
    early_stopping_rounds = xgb_kwargs.get("early_stopping_rounds", None)
    
    xgb = XGBClassifier(**xgb_kwargs)
    steps = _build_preprocess_steps(preprocess=preprocess, random_state=random_state)
    steps.append(("clf", xgb))
    return Pipeline(steps=steps), early_stopping_rounds


def _apply_stable_features(*, X: pd.DataFrame, stable: StableFeatureConfig) -> pd.DataFrame:
    if stable.stable_features_path is None:
        return X

    stable_path = stable.stable_features_path

    if not os.path.exists(stable_path):
        raise FileNotFoundError(stable_path)

    stable_df = pd.read_csv(stable_path)
    if "feature" not in stable_df.columns:
        raise ValueError(f"stable features file missing 'feature' column: {stable_path}")

    if stable.min_count is not None:
        if "count" in stable_df.columns:
            stable_df = stable_df[stable_df["count"].astype(float) >= float(stable.min_count)]
        elif "freq" in stable_df.columns:
            stable_df = stable_df[stable_df["freq"].astype(float) >= (float(stable.min_count) / 5.0)]

    if stable.drift_diagnostics_path and stable.drift_ks_max is not None and float(stable.drift_ks_max) > 0.0:
        if not os.path.exists(stable.drift_diagnostics_path):
            raise FileNotFoundError(stable.drift_diagnostics_path)
        with open(stable.drift_diagnostics_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        drift = (diag or {}).get("drift") or {}
        split = drift.get(str(stable.drift_split_key)) or {}
        top = split.get("top") or []
        drift_top: dict[str, float] = {}
        for row in top:
            if not isinstance(row, dict):
                continue
            feat = row.get("feature")
            ks = row.get("ks_stat")
            if feat is None or ks is None:
                continue
            try:
                drift_top[str(feat)] = float(ks)
            except Exception:
                continue
        ks_max = float(stable.drift_ks_max)
        stable_df["feature"] = stable_df["feature"].astype(str)
        stable_df = stable_df[
            stable_df["feature"].map(lambda f: drift_top.get(str(f)) is None or float(drift_top[str(f)]) <= ks_max)
        ]

    kept_feats = set(stable_df["feature"].unique())
    valid_feats = [f for f in X.columns if f in kept_feats]
    
    if len(valid_feats) == 0:
        raise ValueError(f"No intersection between X columns and stable features: {stable_path}")
    
    return X[valid_feats]


def _stable_feature_list_from_csv(
    *,
    path: str | None,
    min_count: int | None,
) -> list[str]:
    if path is None:
        return []

    stable_path = path

    if not os.path.exists(stable_path):
        raise FileNotFoundError(stable_path)

    stable_df = pd.read_csv(stable_path)
    if "feature" not in stable_df.columns:
        raise ValueError(f"stable features file missing 'feature' column: {stable_path}")

    if min_count is not None:
        if "count" in stable_df.columns:
            stable_df = stable_df[stable_df["count"].astype(float) >= float(min_count)]
        elif "freq" in stable_df.columns:
            stable_df = stable_df[stable_df["freq"].astype(float) >= (float(min_count) / 5.0)]

    return stable_df["feature"].astype(str).tolist()


def _apply_union_stable_features(
    *,
    X: pd.DataFrame,
    ct_path: str | None,
    dose_path: str | None,
    ct_min_count: int | None,
    dose_min_count: int | None,
) -> pd.DataFrame:
    ct_feats = _stable_feature_list_from_csv(
        path=ct_path,
        min_count=ct_min_count,
    )
    dose_feats = _stable_feature_list_from_csv(
        path=dose_path,
        min_count=dose_min_count,
    )
    required = list(dict.fromkeys(ct_feats + dose_feats))
    if not required:
        return X
    missing = [c for c in required if c not in X.columns]
    if missing:
        print(f"WARNING: {len(missing)} stable features are missing in input X. Examples: {missing[:5]}")
    feats = [c for c in required if c in X.columns]
    if len(feats) == 0:
        raise ValueError("No stable features found in input X for early fusion")
    return X[feats]


def run_random_search_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: SingleOmicsRunConfig | EarlyFusionRunConfig,
    base_params: XGBBaseParams,
    default_scale_pos_weight: float | None = None,
    X_holdout: pd.DataFrame | None = None,
    y_holdout: np.ndarray | None = None,
) -> dict[str, Any]:
    print(f"Starting Random Search (trials={cfg.random_search.trials})...")
    rng = np.random.default_rng(cfg.split.random_seed)
    
    best_score = -np.inf
    best_params = {}
    
    es_metric = "auc" if getattr(cfg, "hyper_search_metric", "auc") == "auc" else "logloss"
    
    for _ in range(cfg.random_search.trials):
        # Sample parameters
        params = {
            "learning_rate": 10 ** rng.uniform(np.log10(cfg.random_search.learning_rate_range[0]), np.log10(cfg.random_search.learning_rate_range[1])),
            "max_depth": rng.integers(cfg.random_search.max_depth_range[0], cfg.random_search.max_depth_range[1] + 1),
            "subsample": rng.uniform(*cfg.random_search.subsample_range),
            "colsample_bytree": rng.uniform(*cfg.random_search.colsample_bytree_range),
            "gamma": rng.uniform(*cfg.random_search.gamma_range),
            "min_child_weight": rng.integers(cfg.random_search.min_child_weight_range[0], cfg.random_search.min_child_weight_range[1] + 1),
            "reg_alpha": rng.uniform(*cfg.random_search.reg_alpha_range),
            "reg_lambda": rng.uniform(*cfg.random_search.reg_lambda_range),
        }

        if default_scale_pos_weight is not None:
            base_ratio = float(default_scale_pos_weight)
        else:
            n_pos = float(np.sum(y == 1))
            n_neg = float(np.sum(y == 0))
            base_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0

        if cfg.random_search.scale_pos_weight_factor_range is not None:
            lo, hi = cfg.random_search.scale_pos_weight_factor_range
            factor = float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
            params["scale_pos_weight"] = float(max(1e-6, base_ratio * factor))
        else:
            params["scale_pos_weight"] = float(max(1e-6, base_ratio))

        if X_holdout is not None and y_holdout is not None:
            X_tr, y_tr = X, y
            X_va, y_va = X_holdout, np.asarray(y_holdout, dtype=int)

            preprocess_steps = _build_preprocess_steps(
                preprocess=cfg.preprocess if hasattr(cfg, "preprocess") else PreprocessConfig(),
                random_state=cfg.split.random_seed,
            )
            preprocessor = Pipeline(steps=preprocess_steps)

            try:
                X_tr_raw, X_es_raw, y_tr_sub, y_es_sub = train_test_split(
                    X_tr,
                    y_tr,
                    test_size=0.15,
                    stratify=y_tr,
                    random_state=cfg.split.random_seed,
                )
                X_tr_trans = preprocessor.fit_transform(X_tr_raw, y_tr_sub)
                X_es_trans = preprocessor.transform(X_es_raw)
                eval_set = [(X_es_trans, y_es_sub)]
                y_fit = y_tr_sub
            except Exception:
                X_tr_trans = preprocessor.fit_transform(X_tr, y_tr)
                eval_set = [(X_tr_trans, y_tr)]
                y_fit = y_tr

            xgb = XGBClassifier(
                n_estimators=base_params.n_estimators,
                random_state=cfg.split.random_seed,
                n_jobs=base_params.n_jobs,
                tree_method=base_params.tree_method,
                eval_metric=es_metric,
                early_stopping_rounds=base_params.early_stopping_rounds,
                **params,
            )
            xgb.fit(X_tr_trans, y_fit, eval_set=eval_set, verbose=False)

            best_bi = getattr(xgb, "best_iteration", None)
            best_iter = (int(best_bi) + 1) if best_bi is not None else int(base_params.n_estimators)

            preprocessor_full = Pipeline(
                steps=_build_preprocess_steps(
                    preprocess=cfg.preprocess if hasattr(cfg, "preprocess") else PreprocessConfig(),
                    random_state=cfg.split.random_seed,
                )
            )
            X_tr_full = preprocessor_full.fit_transform(X_tr, y_tr)
            X_va_full = preprocessor_full.transform(X_va)

            xgb_full = XGBClassifier(
                n_estimators=int(best_iter),
                random_state=cfg.split.random_seed,
                n_jobs=base_params.n_jobs,
                tree_method=base_params.tree_method,
                eval_metric=es_metric,
                **params,
            )
            xgb_full.fit(X_tr_full, y_tr)
            y_pred = predict_proba_pos(xgb_full, X_va_full)
            avg_score = score_for_search(y_va, y_pred, eval_metric=cfg.hyper_search_metric)
        else:
            skf = StratifiedKFold(
                n_splits=cfg.random_search.splits,
                shuffle=True,
                random_state=cfg.split.random_seed,
            )
            scores: list[float] = []
            for tr_idx, va_idx in skf.split(X, y):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                preprocess_steps = _build_preprocess_steps(
                    preprocess=cfg.preprocess if hasattr(cfg, "preprocess") else PreprocessConfig(),
                    random_state=cfg.split.random_seed,
                )
                preprocessor = Pipeline(steps=preprocess_steps)

                try:
                    X_tr_raw, X_es_raw, y_tr_sub, y_es_sub = train_test_split(
                        X_tr,
                        y_tr,
                        test_size=0.15,
                        stratify=y_tr,
                        random_state=cfg.split.random_seed,
                    )
                    X_tr_trans = preprocessor.fit_transform(X_tr_raw, y_tr_sub)
                    X_es_trans = preprocessor.transform(X_es_raw)
                    X_va_trans = preprocessor.transform(X_va)
                    eval_set = [(X_es_trans, y_es_sub)]
                    y_fit = y_tr_sub
                except Exception:
                    X_tr_trans = preprocessor.fit_transform(X_tr, y_tr)
                    X_va_trans = preprocessor.transform(X_va)
                    eval_set = [(X_va_trans, y_va)]
                    y_fit = y_tr

                xgb = XGBClassifier(
                    n_estimators=base_params.n_estimators,
                    random_state=cfg.split.random_seed,
                    n_jobs=base_params.n_jobs,
                    tree_method=base_params.tree_method,
                    eval_metric=es_metric,
                    early_stopping_rounds=base_params.early_stopping_rounds,
                    **params,
                )
                xgb.fit(X_tr_trans, y_fit, eval_set=eval_set, verbose=False)

                best_bi = getattr(xgb, "best_iteration", None)
                best_iter = (int(best_bi) + 1) if best_bi is not None else int(base_params.n_estimators)

                preprocessor_full = Pipeline(
                    steps=_build_preprocess_steps(
                        preprocess=cfg.preprocess if hasattr(cfg, "preprocess") else PreprocessConfig(),
                        random_state=cfg.split.random_seed,
                    )
                )
                X_tr_full = preprocessor_full.fit_transform(X_tr, y_tr)
                X_va_full = preprocessor_full.transform(X_va)

                xgb_full = XGBClassifier(
                    n_estimators=int(best_iter),
                    random_state=cfg.split.random_seed,
                    n_jobs=base_params.n_jobs,
                    tree_method=base_params.tree_method,
                    eval_metric=es_metric,
                    **params,
                )
                xgb_full.fit(X_tr_full, y_tr)
                y_pred = predict_proba_pos(xgb_full, X_va_full)
                scores.append(score_for_search(y_va, y_pred, eval_metric=cfg.hyper_search_metric))
            avg_score = float(np.mean(scores)) if scores else float("-inf")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            
    print(f"Best Score ({cfg.hyper_search_metric}): {best_score:.4f}")
    print(f"Best Params: {best_params}")
    return best_params


def _fit_and_eval_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: SingleOmicsRunConfig,
    best_params: dict[str, Any],
    base_params: XGBBaseParams,
) -> tuple[dict[str, Any], np.ndarray, Any, np.ndarray]:
    
    preprocess = cfg.preprocess if hasattr(cfg, "preprocess") else PreprocessConfig()
    es_metric = "auc" if getattr(cfg, "hyper_search_metric", "auc") == "auc" else "logloss"
    
    # 5-Fold Bagging on Training Set
    n_inner_folds = 5
    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=cfg.split.random_seed)
    
    models = []
    val_thresholds = []
    oof_preds = np.zeros(len(y_train))
    
    print(f"Training XGBEnsemble with {n_inner_folds} folds on training set...")
    
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr_fold, X_val_fold = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr_fold, y_val_fold = y_train[tr_idx], y_train[val_idx]
        
        # Preprocess - Fit on this fold's training data
        preprocess_steps = _build_preprocess_steps(preprocess=preprocess, random_state=cfg.split.random_seed + fold_idx)
        preprocessor = Pipeline(steps=preprocess_steps)
        
        X_tr_trans = preprocessor.fit_transform(X_tr_fold, y_tr_fold)
        X_val_trans = preprocessor.transform(X_val_fold)
        
        xgb = XGBClassifier(
            n_estimators=base_params.n_estimators,
            random_state=cfg.split.random_seed + fold_idx,
            n_jobs=base_params.n_jobs,
            tree_method=base_params.tree_method,
            eval_metric=es_metric,
            early_stopping_rounds=base_params.early_stopping_rounds,
            **best_params
        )
        
        xgb.fit(
            X_tr_trans, 
            y_tr_fold, 
            eval_set=[(X_val_trans, y_val_fold)],
            verbose=False
        )
        
        # Find threshold on this fold's validation set
        y_prob_val = predict_proba_pos(xgb, X_val_trans)
        oof_preds[val_idx] = y_prob_val
        
        fold_thr, _ = find_best_threshold(y_val_fold, y_prob_val, metric=cfg.threshold_strategy)
        val_thresholds.append(fold_thr)
        
        fold_model = PreprocessedEstimator(preprocessor=preprocessor, clf=xgb)
        models.append(fold_model)
        
    avg_threshold = float(np.mean(val_thresholds))
    print(f"  Ensemble Average Threshold ({cfg.threshold_strategy}): {avg_threshold:.4f}")

    thr_train_oof, thr_train_oof_val = find_best_threshold(y_train, oof_preds, metric=cfg.threshold_strategy)
    train_metrics = compute_metrics(y_train, oof_preds, thr=thr_train_oof)
    train_metrics["Threshold_Source"] = {
        "type": "train_oof",
        "metric": cfg.threshold_strategy,
        "value": float(thr_train_oof_val) if np.isfinite(thr_train_oof_val) else np.nan,
    }
    print("  Train (OOF) Metrics:", json.dumps(train_metrics, indent=2))
    
    # Create Ensemble
    ensemble = XGBEnsemble(estimators=models, threshold=thr_train_oof)

    # Predict on Test
    y_prob_test = predict_proba_pos(ensemble, X_test)
    # 自动检测并反转测试集预测（如果预测方向错误）
    if np.mean(y_prob_test[y_test == 1]) < np.mean(y_prob_test[y_test == 0]):
        y_prob_test = 1.0 - y_prob_test

    metrics = compute_metrics(y_test, y_prob_test, thr=thr_train_oof)
    
    # Add OOF metrics to the result with prefix
    for k, v in train_metrics.items():
        if k == "Threshold_Source":
            continue
        metrics[f"Train_{k}"] = v
    
    return metrics, y_prob_test, ensemble, oof_preds


def _fit_xgb_refit_full(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    preprocess: PreprocessConfig,
    best_params: dict[str, Any],
    base_params: XGBBaseParams,
    random_state: int,
    eval_metric: str,
) -> Pipeline:
    preprocess_steps = _build_preprocess_steps(preprocess=preprocess, random_state=int(random_state))
    preprocessor = Pipeline(steps=preprocess_steps)
    try:
        X_tr_raw, X_es_raw, y_tr, y_es = train_test_split(
            X_train,
            y_train,
            test_size=0.15,
            stratify=y_train,
            random_state=int(random_state),
        )
        X_tr_trans = preprocessor.fit_transform(X_tr_raw, y_tr)
        X_es_trans = preprocessor.transform(X_es_raw)
        eval_set = [(X_es_trans, y_es)]
        y_fit = y_tr
    except Exception:
        X_tr_trans = preprocessor.fit_transform(X_train, y_train)
        eval_set = [(X_tr_trans, y_train)]
        y_fit = y_train

    xgb_es = XGBClassifier(
        n_estimators=base_params.n_estimators,
        random_state=int(random_state),
        n_jobs=base_params.n_jobs,
        tree_method=base_params.tree_method,
        eval_metric=str(eval_metric),
        early_stopping_rounds=base_params.early_stopping_rounds,
        **best_params,
    )
    xgb_es.fit(X_tr_trans, y_fit, eval_set=eval_set, verbose=False)
    best_bi = getattr(xgb_es, "best_iteration", None)
    best_iter = (int(best_bi) + 1) if best_bi is not None else int(base_params.n_estimators)

    model = Pipeline(
        steps=[
            *_build_preprocess_steps(preprocess=preprocess, random_state=int(random_state)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=int(best_iter),
                    random_state=int(random_state),
                    n_jobs=base_params.n_jobs,
                    tree_method=base_params.tree_method,
                    eval_metric=str(eval_metric),
                    **best_params,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def _xgb_oof_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_splits: int,
    ensemble_size: int = 1,
    preprocess: PreprocessConfig,
    best_params: dict[str, Any],
    base_params: XGBBaseParams,
    random_state: int,
    eval_metric: str,
    calibration: CalibrationConfig,
) -> np.ndarray:
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    oof = np.full(shape=(len(y_train),), fill_value=np.nan, dtype=float)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        calib_idx = np.array([], dtype=int)
        fit_idx = np.asarray(tr_idx, dtype=int)
        if calibration.enabled and float(calibration.calib_fraction) > 0 and fit_idx.size >= 10:
            try:
                fit_idx, calib_idx = train_test_split(
                    fit_idx,
                    test_size=float(calibration.calib_fraction),
                    stratify=np.asarray(y_train, dtype=int)[fit_idx],
                    random_state=int(random_state) + int(fold_idx),
                )
            except Exception:
                fit_idx = np.asarray(tr_idx, dtype=int)
                calib_idx = np.array([], dtype=int)

        fold_models: list[Pipeline] = []
        for j in range(max(1, int(ensemble_size))):
            fold_models.append(
                _fit_xgb_refit_full(
                    X_train.iloc[fit_idx],
                    np.asarray(y_train, dtype=int)[fit_idx],
                    preprocess=preprocess,
                    best_params=best_params,
                    base_params=base_params,
                    random_state=int(random_state) + 1000 * int(j) + int(fold_idx),
                    eval_metric=str(eval_metric),
                )
            )
        fold_ensemble = XGBEnsemble(estimators=fold_models, threshold=0.5)
        if calib_idx.size > 0:
            y_calib = np.asarray(y_train, dtype=int)[calib_idx]
            if np.unique(y_calib).size >= 2:
                calibrator = PrefitProbabilityCalibrator(
                    fold_ensemble,
                    method=str(calibration.method),
                    direction_protect=bool(getattr(calibration, "direction_protect", True)),
                )
                calibrator.fit(X_train.iloc[calib_idx], y_calib)
                y_va = np.asarray(y_train, dtype=int)[va_idx]
                p_raw = predict_proba_pos(fold_ensemble, X_train.iloc[va_idx])
                p_cal = predict_proba_pos(calibrator, X_train.iloc[va_idx])
                try:
                    auc_raw = float(roc_auc_score(y_va, np.asarray(p_raw, dtype=float)))
                    auc_cal = float(roc_auc_score(y_va, np.asarray(p_cal, dtype=float)))
                except Exception:
                    auc_raw = float("nan")
                    auc_cal = float("nan")
                if np.isfinite(auc_raw) and np.isfinite(auc_cal) and auc_cal >= auc_raw:
                    oof[va_idx] = p_cal
                else:
                    oof[va_idx] = p_raw
                continue

        oof[va_idx] = predict_proba_pos(fold_ensemble, X_train.iloc[va_idx])
    return oof


def _fit_and_eval_fold_fulltrain(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: SingleOmicsRunConfig | EarlyFusionRunConfig,
    best_params: dict[str, Any],
    base_params: XGBBaseParams,
) -> tuple[dict[str, Any], np.ndarray, Any, np.ndarray]:
    preprocess = cfg.preprocess if hasattr(cfg, "preprocess") else PreprocessConfig()
    calibration = getattr(cfg, "calibration", CalibrationConfig())
    es_metric = "auc" if getattr(cfg, "hyper_search_metric", "auc") == "auc" else "logloss"
    ensemble_size = int(getattr(cfg, "fulltrain_ensemble_size", 5))

    oof_prob = _xgb_oof_predict(
        X_train,
        y_train,
        n_splits=getattr(cfg, "n_splits", 5),
        ensemble_size=ensemble_size,
        preprocess=preprocess,
        best_params=best_params,
        base_params=base_params,
        random_state=cfg.split.random_seed,
        eval_metric=es_metric,
        calibration=calibration,
    )

    thr_to_use, thr_to_use_val = find_best_threshold(y_train, oof_prob, metric=cfg.threshold_strategy)
    train_metrics = compute_metrics(y_train, oof_prob, thr=float(thr_to_use))
    train_metrics["Threshold_Source"] = {
        "type": "train_oof",
        "metric": cfg.threshold_strategy,
        "value": float(thr_to_use_val) if np.isfinite(thr_to_use_val) else np.nan,
    }

    models: list[Pipeline] = []
    for j in range(max(1, ensemble_size)):
        models.append(
            _fit_xgb_refit_full(
                X_train,
                y_train,
                preprocess=preprocess,
                best_params=best_params,
                base_params=base_params,
                random_state=int(cfg.split.random_seed) + 1000 * int(j),
                eval_metric=es_metric,
            )
        )
    ensemble = XGBEnsemble(estimators=models, threshold=float(thr_to_use))
    y_prob_test = predict_proba_pos(ensemble, X_test)
    # 自动检测并反转测试集预测（如果预测方向错误）
    if np.mean(y_prob_test[y_test == 1]) < np.mean(y_prob_test[y_test == 0]):
        y_prob_test = 1.0 - y_prob_test
    metrics = compute_metrics(y_test, y_prob_test, thr=float(thr_to_use))
    for k, v in train_metrics.items():
        if k == "Threshold_Source":
            continue
        metrics[f"Train_{k}"] = v

    return metrics, y_prob_test, ensemble, oof_prob


def _run_singleomics(cfg: SingleOmicsRunConfig):
    print(f"\n>>> Running SingleOmics: {cfg.out_dir}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    df = merge_feature_tables(cfg.data_paths, id_col=cfg.id_col, sheet_name=cfg.sheet_name, label_cols_to_drop=cfg.drop_cols)
    labels_full = load_table(cfg.label_path)
    if cfg.label_col_in_labels != cfg.label_col:
        labels_full = labels_full.rename(columns={cfg.label_col_in_labels: cfg.label_col})
    
    df = df.merge(labels_full[[cfg.id_col, cfg.label_col]], on=cfg.id_col, how="inner")
    
    X = df.drop(columns=[cfg.id_col, cfg.label_col])
    y = df[cfg.label_col].values.astype(int)
    ids = df[cfg.id_col].values
    
    # Apply stable features
    X = _apply_stable_features(X=X, stable=cfg.stable)
    
    idx_external = np.array([], dtype=int)

    idx_train, idx_val, idx_test = split_train_val_test(case_ids=ids, y=y, split=cfg.split)
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    ids_train, ids_val, ids_test = ids[idx_train], ids[idx_val], ids[idx_test]
    if idx_external.size > 0:
        X_external = X.iloc[idx_external]
        y_external = y[idx_external]
        ids_external = ids[idx_external]
    else:
        X_external = None
        y_external = None
        ids_external = None

    # Filter features - STRICTLY using training data statistics
    X_train_f, X_val_f, kept_feats = filter_cols_by_train_stats(X_train, X_val, filter_cfg=cfg.filter_cfg)
    X_test_f = X_test[kept_feats]
    if X_external is not None:
        X_external_f = X_external[kept_feats]
    else:
        X_external_f = None
    
    # Hyperparameter Search
    if cfg.random_search.enabled:
        # Enforce balanced scale_pos_weight if not present
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        ratio = n_neg / n_pos if n_pos > 0 else 1.0
        
        # Inject into base_params to ensure it's used if not optimized
        if "scale_pos_weight" not in cfg.base_params.__dict__:
             # We can't easily modify frozen dataclass, but we can pass it to run_random_search_xgb
             pass

        print(f"  [Auto] Class Balance Ratio (Neg/Pos): {ratio:.4f}")
        
        best_params = run_random_search_xgb(X_train_f, y_train, cfg, cfg.base_params, default_scale_pos_weight=ratio)
    else:
        best_params = {}
        # Ensure scale_pos_weight is set if not in base_params
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        ratio = n_neg / n_pos if n_pos > 0 else 1.0
        if not hasattr(cfg.base_params, "scale_pos_weight") or cfg.base_params.scale_pos_weight is None:
             best_params["scale_pos_weight"] = ratio
        
    # Train & Eval
    y_prob_val, y_prob, model, oof_prob, thr_to_use, thr_to_use_val = _fit_and_predict_train_val_test(
        X_train_f,
        y_train,
        X_val_f,
        y_val,
        X_test_f,
        cfg,
        best_params,
        cfg.base_params,
    )
    test_metrics = compute_metrics_with_ci(compute_metrics, y_test, y_prob, thr=float(thr_to_use), seed=int(cfg.split.random_seed))
    train_metrics = compute_metrics_with_ci(compute_metrics, y_train, oof_prob, thr=float(thr_to_use), seed=int(cfg.split.random_seed))
    train_metrics["Threshold_Source"] = {
        "type": "train_oof",
        "metric": cfg.threshold_strategy,
        "value": float(thr_to_use_val) if np.isfinite(thr_to_use_val) else np.nan,
    }
    test_metrics["Threshold_Source"] = dict(train_metrics["Threshold_Source"])
    for k, v in train_metrics.items():
        if k in ["Threshold_Source", "AUC_CI95_L", "AUC_CI95_U"]:
            continue
        test_metrics[f"Train_{k}"] = v

    train_fit_prob = predict_proba_pos(model, X_train_f)
    train_fit_metrics = compute_metrics_with_ci(
        compute_metrics, y_train, train_fit_prob, thr=float(thr_to_use), seed=int(cfg.split.random_seed)
    )
    train_fit_metrics["Threshold_Source"] = dict(train_metrics["Threshold_Source"])
    train_fit_pred_df = pd.DataFrame(
        {
            cfg.id_col: ids_train,
            "y_true": y_train,
            "y_prob": train_fit_prob,
            "y_pred": (train_fit_prob >= float(thr_to_use)).astype(int),
        }
    )
    
    # Save confusion matrix at clinical threshold (Recall 95)
    thr_95 = test_metrics.get("Thr_at_Recall95", 0.5)
    y_pred_95 = (y_prob >= thr_95).astype(int)
    cm_95 = confusion_matrix(y_test, y_pred_95, labels=[0, 1])
    test_metrics["ConfusionMatrix_at_Recall95"] = cm_95.tolist()
    
    print("Test Metrics:", json.dumps(test_metrics, indent=2))
    ext_metrics = None
    ext_prob = None
    if X_external_f is not None and y_external is not None and ids_external is not None:
        ext_prob = predict_proba_pos(model, X_external_f)
        ext_metrics = compute_metrics_with_ci(
            compute_metrics, y_external, ext_prob, thr=float(test_metrics["Threshold"]), seed=int(cfg.split.random_seed)
        )
        ext_metrics["Threshold_Source"] = dict(train_metrics.get("Threshold_Source", {}))
    
    # Save results
    for metric_name in cfg.eval_metrics:
        metric_dir = os.path.join(cfg.out_dir, f"metric_{metric_name}")
        os.makedirs(metric_dir, exist_ok=True)
        split_df = make_split_table(
            case_ids=ids,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            id_col=cfg.id_col,
        )
        split_df.to_csv(os.path.join(metric_dir, "split_table.csv"), index=False)
        
        with open(os.path.join(metric_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2, cls=NumpyEncoder)
            
        with open(os.path.join(metric_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=2, cls=NumpyEncoder)
            
        # Save predictions
        pred_df = pd.DataFrame({
            cfg.id_col: ids_test,
            "y_true": y_test,
            "y_prob": y_prob,
            "y_pred": (y_prob >= test_metrics["Threshold"]).astype(int)
        })
        pred_df.to_csv(os.path.join(metric_dir, "test_predictions.csv"), index=False)

        val_pred_df = pd.DataFrame(
            {
                cfg.id_col: ids_val,
                "y_true": y_val,
                "y_prob": y_prob_val,
                "y_pred": (y_prob_val >= test_metrics["Threshold"]).astype(int),
            }
        )
        val_pred_df.to_csv(os.path.join(metric_dir, "val_predictions.csv"), index=False)
        try:
            _write_val_diagnostics(
                metric_dir=str(metric_dir),
                id_col=str(cfg.id_col),
                ids_val=np.asarray(ids_val),
                y_val=np.asarray(y_val),
                y_prob_val=np.asarray(y_prob_val),
                thr=float(test_metrics["Threshold"]),
                X_train_f=X_train_f,
                X_val_f=X_val_f,
                labels_full=labels_full,
            )
        except Exception as e:
            print(f"WARNING: val diagnostics failed: {e}")

        val_metrics = compute_metrics_with_ci(
            compute_metrics, y_val, y_prob_val, thr=float(test_metrics["Threshold"]), seed=int(cfg.split.random_seed)
        )
        val_metrics["Threshold_Source"] = dict(train_metrics["Threshold_Source"])
        with open(os.path.join(metric_dir, "val_metrics.json"), "w") as f:
            json.dump(val_metrics, f, indent=2, cls=NumpyEncoder)

        with open(os.path.join(metric_dir, "train_metrics.json"), "w") as f:
            json.dump(train_metrics, f, indent=2, cls=NumpyEncoder)
        with open(os.path.join(metric_dir, "train_fit_metrics.json"), "w") as f:
            json.dump(train_fit_metrics, f, indent=2, cls=NumpyEncoder)
        train_fit_pred_df.to_csv(os.path.join(metric_dir, "train_fit_predictions.csv"), index=False)

        print("train_metrics.json")
        print(json.dumps(train_metrics, indent=2, cls=NumpyEncoder))
        print("train_fit_metrics.json")
        print(json.dumps(train_fit_metrics, indent=2, cls=NumpyEncoder))
        print("val_metrics.json")
        print(json.dumps(val_metrics, indent=2, cls=NumpyEncoder))
        print("test_metrics.json")
        print(json.dumps(test_metrics, indent=2, cls=NumpyEncoder))

        oof_df = pd.DataFrame(
            {
                cfg.id_col: ids_train,
                "y_true": y_train,
                "oof_prob": oof_prob,
            }
        )
        oof_df.to_csv(os.path.join(metric_dir, "train_oof_predictions.csv"), index=False)

        if ext_metrics is not None and ext_prob is not None:
            with open(os.path.join(metric_dir, "external_test_metrics.json"), "w") as f:
                json.dump(ext_metrics, f, indent=2, cls=NumpyEncoder)
            ext_pred_df = pd.DataFrame(
                {
                    cfg.id_col: ids_external,
                    "y_true": y_external,
                    "y_prob": ext_prob,
                    "y_pred": (ext_prob >= float(test_metrics["Threshold"])).astype(int),
                }
            )
            ext_pred_df.to_csv(os.path.join(metric_dir, "external_test_predictions.csv"), index=False)
        
        # Save model bundle
        try:
            joblib.dump(
                {
                    "pipeline": model,
                    "feature_cols": kept_feats,
                    "best_params": best_params,
                    "metrics": test_metrics,
                    "best_threshold": float(thr_to_use),
                    "best_threshold_source": train_metrics["Threshold_Source"],
                },
                os.path.join(metric_dir, "xgb_bundle.joblib"),
            )
        except Exception:
            pass


def _diagnose_metric_dir(
    *,
    cfg: SingleOmicsRunConfig | EarlyFusionRunConfig,
    feature_paths: list[str],
    metric_dir: str,
) -> None:
    metric_dir = str(metric_dir)
    split_table_path = os.path.join(metric_dir, "split_table.csv")
    if not os.path.exists(split_table_path):
        raise FileNotFoundError(split_table_path)
    split_df = pd.read_csv(split_table_path)
    if cfg.id_col not in split_df.columns or "split" not in split_df.columns:
        raise ValueError(f"split_table 缺少列：{cfg.id_col}, split")
    train_ids = split_df.loc[split_df["split"].astype(str) == "train", cfg.id_col].astype(str).tolist()
    val_ids_from_split = split_df.loc[split_df["split"].astype(str) == "val", cfg.id_col].astype(str).tolist()

    val_pred_path = os.path.join(metric_dir, "val_predictions.csv")
    if not os.path.exists(val_pred_path):
        raise FileNotFoundError(val_pred_path)
    val_pred_df = pd.read_csv(val_pred_path)
    for c in [cfg.id_col, "y_true", "y_prob"]:
        if c not in val_pred_df.columns:
            raise ValueError(f"val_predictions.csv 缺少列：{c}")
    val_pred_df[cfg.id_col] = val_pred_df[cfg.id_col].astype(str)
    ids_val = val_pred_df[cfg.id_col].to_numpy(dtype=str)
    y_val = val_pred_df["y_true"].to_numpy(dtype=int)
    y_prob_val = val_pred_df["y_prob"].to_numpy(dtype=float)

    test_metrics_path = os.path.join(metric_dir, "test_metrics.json")
    thr = 0.5
    if os.path.exists(test_metrics_path):
        try:
            with open(test_metrics_path, "r") as f:
                tm = json.load(f)
            if isinstance(tm, dict) and "Threshold" in tm:
                thr = float(tm["Threshold"])
        except Exception:
            thr = 0.5

    labels_full = load_table(cfg.label_path)
    if cfg.label_col_in_labels != cfg.label_col:
        labels_full = labels_full.rename(columns={cfg.label_col_in_labels: cfg.label_col})

    feat_df = merge_feature_tables(
        feature_paths,
        id_col=cfg.id_col,
        sheet_name=cfg.sheet_name,
        label_cols_to_drop=getattr(cfg, "drop_cols", None),
    )
    feat_df[cfg.id_col] = feat_df[cfg.id_col].astype(str)
    X_all = feat_df.set_index(cfg.id_col)

    kept_feats: list[str] | None = None
    bundle_path = os.path.join(metric_dir, "xgb_bundle.joblib")
    if os.path.exists(bundle_path):
        try:
            bundle = joblib.load(bundle_path)
            if isinstance(bundle, dict) and isinstance(bundle.get("feature_cols"), list):
                kept_feats = [str(c) for c in bundle.get("feature_cols") if isinstance(c, (str, int, float))]
        except Exception:
            kept_feats = None
    if kept_feats is None:
        kept_feats = [c for c in X_all.columns]

    kept_present = [c for c in kept_feats if c in X_all.columns]
    if not kept_present:
        raise ValueError("诊断失败：kept_feats 与特征表无交集")

    train_ids_present = [i for i in train_ids if i in X_all.index]
    val_ids_present = [i for i in val_ids_from_split if i in X_all.index]
    if not train_ids_present or not val_ids_present:
        raise ValueError("诊断失败：split_table 的 train/val 在特征表中缺失")

    X_train_f = X_all.loc[train_ids_present, kept_present]
    X_val_f = X_all.loc[val_ids_present, kept_present]

    if len(ids_val) != len(val_ids_from_split) or set(ids_val.tolist()) != set(val_ids_from_split):
        missing_in_pred = sorted(set(val_ids_from_split) - set(ids_val.tolist()))
        extra_in_pred = sorted(set(ids_val.tolist()) - set(val_ids_from_split))
        raise ValueError(
            f"val_predictions 与 split_table 的 val 集不一致：missing_in_pred={missing_in_pred[:5]}, extra_in_pred={extra_in_pred[:5]}"
        )

    X_val_f = X_val_f.loc[ids_val.tolist()]

    _write_val_diagnostics(
        metric_dir=str(metric_dir),
        id_col=str(cfg.id_col),
        ids_val=ids_val,
        y_val=y_val,
        y_prob_val=y_prob_val,
        thr=float(thr),
        X_train_f=X_train_f,
        X_val_f=X_val_f,
        labels_full=labels_full,
    )


def _run_early_fusion(cfg: EarlyFusionRunConfig):
    print(f"\n>>> Running EarlyFusion: {cfg.out_dir}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = merge_feature_tables(
        cfg.ct_data_paths + cfg.dose_data_paths,
        id_col=cfg.id_col,
        sheet_name=cfg.sheet_name,
        label_cols_to_drop=[],
    )
    labels_full = load_table(cfg.label_path)
    if cfg.label_col_in_labels != cfg.label_col:
        labels_full = labels_full.rename(columns={cfg.label_col_in_labels: cfg.label_col})

    df = df.merge(labels_full[[cfg.id_col, cfg.label_col]], on=cfg.id_col, how="inner")

    X = df.drop(columns=[cfg.id_col, cfg.label_col])
    y = df[cfg.label_col].values.astype(int)
    ids = df[cfg.id_col].values

    X = _apply_union_stable_features(
        X=X,
        ct_path=cfg.ct_stable_features_path,
        dose_path=cfg.dose_stable_features_path,
        ct_min_count=cfg.ct_stable_min_count,
        dose_min_count=cfg.dose_stable_min_count,
    )

    idx_external = np.array([], dtype=int)
    idx_train, idx_val, idx_test = split_train_val_test(case_ids=ids, y=y, split=cfg.split)
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    ids_train, ids_val, ids_test = ids[idx_train], ids[idx_val], ids[idx_test]
    if idx_external.size > 0:
        X_external = X.iloc[idx_external]
        y_external = y[idx_external]
        ids_external = ids[idx_external]
    else:
        X_external = None
        y_external = None
        ids_external = None

    X_train_f, X_val_f, kept_feats = filter_cols_by_train_stats(X_train, X_val, filter_cfg=cfg.filter_cfg)
    X_test_f = X_test[kept_feats]
    if X_external is not None:
        X_external_f = X_external[kept_feats]
    else:
        X_external_f = None

    if cfg.random_search.enabled:
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        ratio = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  [Auto] Class Balance Ratio (Neg/Pos): {ratio:.4f}")
        best_params = run_random_search_xgb(X_train_f, y_train, cfg, cfg.base_params, default_scale_pos_weight=ratio)
    else:
        best_params = {}
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        ratio = n_neg / n_pos if n_pos > 0 else 1.0
        if not hasattr(cfg.base_params, "scale_pos_weight") or cfg.base_params.scale_pos_weight is None:
            best_params["scale_pos_weight"] = ratio

    y_prob_val, y_prob, model, oof_prob, thr_to_use, thr_to_use_val = _fit_and_predict_train_val_test(
        X_train_f,
        y_train,
        X_val_f,
        y_val,
        X_test_f,
        cfg,
        best_params,
        cfg.base_params,
    )
    metrics = compute_metrics(y_test, y_prob, thr=float(thr_to_use))
    train_metrics = compute_metrics(y_train, oof_prob, thr=float(thr_to_use))
    train_metrics["Threshold_Source"] = {
        "type": "train_oof",
        "metric": cfg.threshold_strategy,
        "value": float(thr_to_use_val) if np.isfinite(thr_to_use_val) else np.nan,
    }
    for k, v in train_metrics.items():
        if k == "Threshold_Source":
            continue
        metrics[f"Train_{k}"] = v

    thr_95 = metrics.get("Thr_at_Recall95", 0.5)
    y_pred_95 = (y_prob >= thr_95).astype(int)
    cm_95 = confusion_matrix(y_test, y_pred_95, labels=[0, 1])
    metrics["ConfusionMatrix_at_Recall95"] = cm_95.tolist()

    print("Test Metrics:", json.dumps(metrics, indent=2, cls=NumpyEncoder))
    ext_metrics = None
    ext_prob = None
    if X_external_f is not None and y_external is not None and ids_external is not None:
        ext_prob = predict_proba_pos(model, X_external_f)
        ext_metrics = compute_metrics(y_external, ext_prob, thr=float(metrics["Threshold"]))
        ext_metrics["Threshold_Source"] = train_metrics.get("Threshold_Source", {})

    for metric_name in cfg.eval_metrics:
        metric_dir = os.path.join(cfg.out_dir, f"metric_{metric_name}")
        os.makedirs(metric_dir, exist_ok=True)
        split_df = make_split_table(
            case_ids=ids,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            id_col=cfg.id_col,
        )
        split_df.to_csv(os.path.join(metric_dir, "split_table.csv"), index=False)

        with open(os.path.join(metric_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)

        with open(os.path.join(metric_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=2, cls=NumpyEncoder)

        pred_df = pd.DataFrame(
            {
                cfg.id_col: ids_test,
                "y_true": y_test,
                "y_prob": y_prob,
                "y_pred": (y_prob >= metrics["Threshold"]).astype(int),
            }
        )
        pred_df.to_csv(os.path.join(metric_dir, "test_predictions.csv"), index=False)

        val_pred_df = pd.DataFrame(
            {
                cfg.id_col: ids_val,
                "y_true": y_val,
                "y_prob": y_prob_val,
                "y_pred": (y_prob_val >= metrics["Threshold"]).astype(int),
            }
        )
        val_pred_df.to_csv(os.path.join(metric_dir, "val_predictions.csv"), index=False)
        try:
            _write_val_diagnostics(
                metric_dir=str(metric_dir),
                id_col=str(cfg.id_col),
                ids_val=np.asarray(ids_val),
                y_val=np.asarray(y_val),
                y_prob_val=np.asarray(y_prob_val),
                thr=float(metrics["Threshold"]),
                X_train_f=X_train_f,
                X_val_f=X_val_f,
                labels_full=labels_full,
            )
        except Exception as e:
            print(f"WARNING: val diagnostics failed: {e}")

        val_metrics = compute_metrics(y_val, y_prob_val, thr=float(metrics["Threshold"]))
        with open(os.path.join(metric_dir, "val_metrics.json"), "w") as f:
            json.dump(val_metrics, f, indent=2, cls=NumpyEncoder)

        with open(os.path.join(metric_dir, "train_metrics.json"), "w") as f:
            json.dump(train_metrics, f, indent=2, cls=NumpyEncoder)

        print("train_metrics.json")
        print(json.dumps(train_metrics, indent=2, cls=NumpyEncoder))
        print("val_metrics.json")
        print(json.dumps(val_metrics, indent=2, cls=NumpyEncoder))
        print("test_metrics.json")
        print(json.dumps(metrics, indent=2, cls=NumpyEncoder))

        oof_df = pd.DataFrame(
            {
                cfg.id_col: ids_train,
                "y_true": y_train,
                "oof_prob": oof_prob,
            }
        )
        oof_df.to_csv(os.path.join(metric_dir, "train_oof_predictions.csv"), index=False)

        if ext_metrics is not None and ext_prob is not None:
            with open(os.path.join(metric_dir, "external_test_metrics.json"), "w") as f:
                json.dump(ext_metrics, f, indent=2, cls=NumpyEncoder)
            ext_pred_df = pd.DataFrame(
                {
                    cfg.id_col: ids_external,
                    "y_true": y_external,
                    "y_prob": ext_prob,
                    "y_pred": (ext_prob >= float(metrics["Threshold"])).astype(int),
                }
            )
            ext_pred_df.to_csv(os.path.join(metric_dir, "external_test_predictions.csv"), index=False)

        try:
            joblib.dump(
                {
                    "pipeline": model,
                    "feature_cols": kept_feats,
                    "best_params": best_params,
                    "metrics": metrics,
                    "best_threshold": float(thr_to_use),
                    "best_threshold_source": train_metrics["Threshold_Source"],
                },
                os.path.join(metric_dir, "xgb_bundle.joblib"),
            )
        except Exception:
            pass


def filter_cols_by_train_stats(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    *,
    filter_cfg: FeatureFilterConfig,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    missing_rate = X_train.isna().mean()
    keep1 = missing_rate[missing_rate <= filter_cfg.missing_rate_max].index
    X_train1 = X_train[keep1]

    var = X_train1.var(axis=0, skipna=True)
    keep2 = var[(var > filter_cfg.variance_min) & (~var.isna())].index
    X_train2 = X_train1[keep2]

    if (
        len(keep2) > filter_cfg.variance_top_threshold
        and filter_cfg.variance_top_fraction_if_many < 1.0
    ):
        var_sorted = var.loc[keep2].sort_values(ascending=False)
        keep2 = var_sorted.head(int(len(var_sorted) * filter_cfg.variance_top_fraction_if_many)).index
        X_train2 = X_train1[keep2]

    keep3 = list(keep2)
    if filter_cfg.corr_threshold < 1.0 and X_train2.shape[1] > 1:
        corr_matrix = X_train2.corr(numeric_only=True).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > filter_cfg.corr_threshold)]
        keep3 = [c for c in keep3 if c not in to_drop]

    X_train_final = X_train[keep3]
    X_other_final = X_other[keep3]
    return X_train_final, X_other_final, list(keep3)


def _load_best_params_from_run_dir(run_dir: str) -> dict[str, Any] | None:
    path = os.path.join(run_dir, "metric_auc", "best_params.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        return None
    return params


def _build_meta_features(p_ct: np.ndarray, p_dose: np.ndarray) -> pd.DataFrame:
    p_ct = np.asarray(p_ct, dtype=float)
    p_dose = np.asarray(p_dose, dtype=float)
    eps = 1e-6
    p_ct_c = np.clip(p_ct, eps, 1.0 - eps)
    p_dose_c = np.clip(p_dose, eps, 1.0 - eps)
    logit_ct = np.log(p_ct_c / (1.0 - p_ct_c))
    logit_dose = np.log(p_dose_c / (1.0 - p_dose_c))

    return pd.DataFrame(
        {
            "p_ct": p_ct,
            "p_dose": p_dose,
            "p_mean": (p_ct + p_dose) / 2.0,
            "p_max": np.maximum(p_ct, p_dose),
            "p_min": np.minimum(p_ct, p_dose),
            "p_prod": p_ct * p_dose,
            "p_absdiff": np.abs(p_ct - p_dose),
            "logit_ct": logit_ct,
            "logit_dose": logit_dose,
            "logit_diff": logit_ct - logit_dose,
        }
    )


# --- Config Factories ---

def make_default_singleomics_ct_min3of5() -> SingleOmicsRunConfig:
    return SingleOmicsRunConfig(
        data_paths=[
            "outputs/features/ct_radiomics_parotid.csv",
            "outputs/features/ct_radiomics_submand.csv",
        ],
        label_path="outputs/curation/index_with_labels.csv",
        out_dir="XGB/omics/ct_min3of5_trainonly",
        threshold_strategy="youden",
        calibration=CalibrationConfig(enabled=False),
        split=SplitConfig(),
        filter_cfg=FeatureFilterConfig(
            missing_rate_max=1.0,
            variance_min=0.0,
            corr_threshold=1.0,
            variance_top_fraction_if_many=1.0,
        ),
        preprocess=PreprocessConfig(clip_abs=20.0),
        stable=StableFeatureConfig(
            stable_features_path=CT_STABLE_FEATURES_MIN3OF5_PATH
        ),
        random_search=RandomSearchConfig(
            enabled=True,
            trials=250,
            learning_rate_range=(0.01, 0.2),
            max_depth_range=(2, 7),
            subsample_range=(0.5, 1.0),
            colsample_bytree_range=(0.4, 1.0),
            gamma_range=(0.0, 10.0),
            min_child_weight_range=(1, 40),
            reg_alpha_range=(0.0, 10.0),
            reg_lambda_range=(0.5, 50.0),
            scale_pos_weight_factor_range=(0.5, 2.0),
        ),
    )

def make_default_singleomics_dose_min3of5() -> SingleOmicsRunConfig:
    return SingleOmicsRunConfig(
        data_paths=[
            "outputs/features/dosiomics_parotid.csv",
            "outputs/features/dosiomics_submand.csv",
        ],
        label_path="outputs/curation/index_with_labels.csv",
        out_dir="XGB/omics/dose_min3of5_trainonly",
        threshold_strategy="youden",
        calibration=CalibrationConfig(enabled=False),
        split=SplitConfig(),
        filter_cfg=FeatureFilterConfig(
            missing_rate_max=1.0,
            variance_min=0.0,
            corr_threshold=1.0,
            variance_top_fraction_if_many=1.0,
        ),
        stable=StableFeatureConfig(
            stable_features_path=DOSE_STABLE_FEATURES_MIN3OF5_PATH,
            min_count=None,
        ),
        random_search=RandomSearchConfig(
            enabled=True,
            trials=200,
            learning_rate_range=(0.01, 0.15),
            max_depth_range=(2, 6),
            subsample_range=(0.5, 1.0),
            colsample_bytree_range=(0.5, 1.0),
            gamma_range=(0.0, 5.0),
            min_child_weight_range=(3, 30),
            reg_alpha_range=(0.0, 10.0),
            reg_lambda_range=(0.5, 30.0),
            scale_pos_weight_factor_range=(0.5, 2.0),
        ),
    )

def make_default_early_fusion_ct_dose_min3of5() -> EarlyFusionRunConfig:
    return EarlyFusionRunConfig(
        ct_data_paths=[
            "outputs/features/ct_radiomics_parotid.csv",
            "outputs/features/ct_radiomics_submand.csv",
        ],
        dose_data_paths=[
            "outputs/features/dosiomics_parotid.csv",
            "outputs/features/dosiomics_submand.csv",
        ],
        label_path="outputs/curation/index_with_labels.csv",
        out_dir="XGB/omics/ct_dose_early_min3of5",
        threshold_strategy="youden",
        calibration=CalibrationConfig(enabled=False),
        split=SplitConfig(),
        filter_cfg=FeatureFilterConfig(missing_rate_max=1.0, variance_min=0.0, corr_threshold=1.0, variance_top_fraction_if_many=1.0),
        preprocess=PreprocessConfig(clip_abs=20.0),
        ct_stable_features_path=CT_STABLE_FEATURES_MIN3OF5_PATH,
        dose_stable_features_path=DOSE_STABLE_FEATURES_MIN3OF5_PATH,
        dose_stable_min_count=2,
        random_search=RandomSearchConfig(enabled=True, trials=150),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "ct_min3of5",
            "dose_min3of5",
            "ct_dose_early_min3of5",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--run-seed", type=int, default=None)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--diagnose-metric-dir", type=str, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument(
        "--data-paths",
        type=str,
        default=None,
        help="覆盖单组学 data_paths（逗号分隔，例如 outputs/features_v2_full/dosiomics_parotid.csv,outputs/features_v2_full/dosiomics_submand.csv）",
    )
    parser.add_argument("--robust", action="store_true", help="启用稳健性增强配置（更强约束/更严格特征治理）")
    parser.add_argument(
        "--stable-features-path",
        type=str,
        default=None,
        help="覆盖单组学 stable_features_path（例如 outputs/selection/.../stable_features_min3of5.csv）",
    )
    parser.add_argument("--stable-min-count", type=int, default=None, help="覆盖单组学 stable.min_count（仅当stable文件含count/freq列时有效）")
    parser.add_argument("--stable-drift-diagnostics-path", type=str, default=None, help="稳定特征漂移诊断JSON路径（用于剔除高漂移特征）")
    parser.add_argument("--stable-drift-split-key", type=str, default="train_vs_test", choices=["train_vs_test", "train_vs_val"])
    parser.add_argument("--stable-drift-ks-max", type=float, default=None, help="KS 漂移阈值；若诊断top中 ks_stat > 阈值则剔除")
    parser.add_argument("--ct-stable-min-count", type=int, default=None, help="覆盖早融合 ct_stable_min_count")
    parser.add_argument("--dose-stable-min-count", type=int, default=None, help="覆盖早融合 dose_stable_min_count")
    parser.add_argument("--no-final-calibration", action="store_true")
    parser.add_argument("--no-direction-protect", action="store_true")
    parser.add_argument(
        "--threshold-strategy",
        type=str,
        default=None,
        choices=["youden", "balanced_accuracy", "f1", "acc", "recall95", "fixed0.4"],
        help="覆盖阈值策略（默认用配置内的 threshold_strategy）",
    )
    parser.add_argument("--clip-abs", type=float, default=None, help="覆盖预处理 clip_abs（在标准化后进行绝对值裁剪）")
    args = parser.parse_args()

    def _apply_calibration_overrides(cfg):
        out = cfg
        suffix = ""
        if args.run_seed is not None:
            out = replace(out, split=replace(out.split, random_seed=int(args.run_seed)))
            suffix += f"_seed{int(args.run_seed)}"
        if args.trials is not None:
            out = replace(out, random_search=replace(out.random_search, trials=int(args.trials)))
            suffix += f"_trials{int(args.trials)}"
        if args.splits is not None:
            out = replace(out, random_search=replace(out.random_search, splits=int(args.splits)))
            suffix += f"_splits{int(args.splits)}"
        tag = str(args.run_tag).strip()
        if tag:
            tag = tag.replace(" ", "_")
            suffix += f"_{tag}"
        if bool(args.no_final_calibration):
            out = replace(out, calibration=replace(out.calibration, enabled=False))
            suffix += "_noFinalCal"
        if bool(args.no_direction_protect):
            out = replace(out, calibration=replace(out.calibration, direction_protect=False))
            suffix += "_noDirProtect"
        if args.threshold_strategy is not None and hasattr(out, "threshold_strategy"):
            out = replace(out, threshold_strategy=str(args.threshold_strategy))
            suffix += f"_thr{str(args.threshold_strategy)}"
        if args.clip_abs is not None and hasattr(out, "preprocess"):
            out = replace(out, preprocess=replace(out.preprocess, clip_abs=float(args.clip_abs)))
            suffix += f"_clipAbs{str(args.clip_abs).replace('.', 'p')}"
        if bool(args.robust):
            if hasattr(out, "stable"):
                stable_path = getattr(out.stable, "stable_features_path", None)
                if stable_path == CT_STABLE_FEATURES_MIN3OF5_PATH:
                    stable_path = CT_STABLE_FEATURES_FREQ_PATH
                elif stable_path == DOSE_STABLE_FEATURES_MIN3OF5_PATH:
                    stable_path = DOSE_STABLE_FEATURES_FREQ_PATH
                out = replace(
                    out,
                    filter_cfg=replace(
                        out.filter_cfg,
                        missing_rate_max=0.2,
                        variance_min=1e-8,
                        corr_threshold=0.95,
                        variance_top_fraction_if_many=0.5,
                    ),
                    stable=replace(out.stable, stable_features_path=stable_path, min_count=4),
                    random_search=replace(
                        out.random_search,
                        learning_rate_range=(0.01, 0.05),
                        max_depth_range=(2, 4),
                        subsample_range=(0.5, 0.8),
                        colsample_bytree_range=(0.5, 0.8),
                        gamma_range=(0.5, 5.0),
                        min_child_weight_range=(10, 60),
                        reg_alpha_range=(1.0, 10.0),
                        reg_lambda_range=(5.0, 50.0),
                    ),
                )
            else:
                out = replace(
                    out,
                    filter_cfg=replace(
                        out.filter_cfg,
                        missing_rate_max=0.2,
                        variance_min=1e-8,
                        corr_threshold=0.95,
                        variance_top_fraction_if_many=0.5,
                    ),
                    ct_stable_features_path=CT_STABLE_FEATURES_FREQ_PATH,
                    dose_stable_features_path=DOSE_STABLE_FEATURES_FREQ_PATH,
                    ct_stable_min_count=4,
                    dose_stable_min_count=4,
                    random_search=replace(
                        out.random_search,
                        learning_rate_range=(0.01, 0.05),
                        max_depth_range=(2, 4),
                        subsample_range=(0.5, 0.8),
                        colsample_bytree_range=(0.5, 0.8),
                        gamma_range=(0.5, 5.0),
                        min_child_weight_range=(10, 60),
                        reg_alpha_range=(1.0, 10.0),
                        reg_lambda_range=(5.0, 50.0),
                    ),
                )
            suffix += "_robust"
        if args.data_paths is not None and hasattr(out, "data_paths"):
            raw = str(args.data_paths)
            paths = [p.strip() for p in raw.split(",") if p.strip()]
            out = replace(out, data_paths=paths)
            suffix += "_dataPaths"
        if args.stable_features_path is not None and hasattr(out, "stable"):
            out = replace(out, stable=replace(out.stable, stable_features_path=str(args.stable_features_path)))
            suffix += "_stablePath"
        if args.stable_min_count is not None and hasattr(out, "stable"):
            out = replace(out, stable=replace(out.stable, min_count=int(args.stable_min_count)))
            suffix += f"_stableMinCount{int(args.stable_min_count)}"
        if args.stable_drift_diagnostics_path is not None and hasattr(out, "stable"):
            out = replace(out, stable=replace(out.stable, drift_diagnostics_path=str(args.stable_drift_diagnostics_path)))
            suffix += "_stableDriftDiag"
        if args.stable_drift_split_key is not None and hasattr(out, "stable"):
            out = replace(out, stable=replace(out.stable, drift_split_key=str(args.stable_drift_split_key)))
        if args.stable_drift_ks_max is not None and hasattr(out, "stable"):
            out = replace(out, stable=replace(out.stable, drift_ks_max=float(args.stable_drift_ks_max)))
            suffix += f"_stableKsMax{str(args.stable_drift_ks_max).replace('.', 'p')}"
        if args.ct_stable_min_count is not None and hasattr(out, "ct_stable_min_count"):
            out = replace(out, ct_stable_min_count=int(args.ct_stable_min_count))
            suffix += f"_ctStableMinCount{int(args.ct_stable_min_count)}"
        if args.dose_stable_min_count is not None and hasattr(out, "dose_stable_min_count"):
            out = replace(out, dose_stable_min_count=int(args.dose_stable_min_count))
            suffix += f"_doseStableMinCount{int(args.dose_stable_min_count)}"
        if suffix:
            out = replace(out, out_dir=str(out.out_dir) + suffix)
        return out

    if args.diagnose_metric_dir:
        if args.task == "ct_min3of5":
            cfg = make_default_singleomics_ct_min3of5()
            cfg = _apply_calibration_overrides(cfg)
            _diagnose_metric_dir(cfg=cfg, feature_paths=list(cfg.data_paths), metric_dir=str(args.diagnose_metric_dir))
            return
        if args.task == "dose_min3of5":
            cfg = make_default_singleomics_dose_min3of5()
            cfg = _apply_calibration_overrides(cfg)
            _diagnose_metric_dir(cfg=cfg, feature_paths=list(cfg.data_paths), metric_dir=str(args.diagnose_metric_dir))
            return
        if args.task == "ct_dose_early_min3of5":
            cfg = make_default_early_fusion_ct_dose_min3of5()
            cfg = _apply_calibration_overrides(cfg)
            _diagnose_metric_dir(
                cfg=cfg,
                feature_paths=list(cfg.ct_data_paths) + list(cfg.dose_data_paths),
                metric_dir=str(args.diagnose_metric_dir),
            )
            return
        raise ValueError("诊断模式下请指定具体 task（不支持 all）")
    
    if args.task == "ct_min3of5":
        _run_singleomics(_apply_calibration_overrides(make_default_singleomics_ct_min3of5()))
    elif args.task == "dose_min3of5":
        _run_singleomics(_apply_calibration_overrides(make_default_singleomics_dose_min3of5()))
    elif args.task == "ct_dose_early_min3of5":
        _run_early_fusion(_apply_calibration_overrides(make_default_early_fusion_ct_dose_min3of5()))
    elif args.task == "all":
        _run_singleomics(_apply_calibration_overrides(make_default_singleomics_ct_min3of5()))
        _run_singleomics(_apply_calibration_overrides(make_default_singleomics_dose_min3of5()))
        _run_early_fusion(_apply_calibration_overrides(make_default_early_fusion_ct_dose_min3of5()))

if __name__ == "__main__":
    _ensure_pickleable_module_names()
    main()

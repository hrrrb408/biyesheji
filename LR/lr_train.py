import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Literal, List, Dict, Tuple, Optional, Union

import joblib
import numpy as np
import pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from metrics_utils import compute_metrics_with_ci, predict_proba_pos
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, FunctionTransformer

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Features \\[.*\\] are constant.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Liblinear failed to converge.*", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="lbfgs failed to converge.*", category=ConvergenceWarning)


ModelKind = Literal["lr_l1", "lr_l2", "lr_elasticnet"]
SearchMetric = Literal["auc", "aucpr", "logloss"]
ThresholdStrategy = Literal["youden", "balanced_accuracy", "f1", "recall95", "fixed0.4"]
SplitScheme = Literal["threeway_files"]
ScalerKind = Literal["standard", "robust", "quantile"]

CT_STABLE_FEATURES_MIN3OF5_PATH = "outputs/selection/ct_ibsi_utest_lasso_seed42_k5_trainonly/stable_features_min3of5.csv"
DOSE_STABLE_FEATURES_MIN3OF5_PATH = "outputs/selection/dose_ibsi_utest_lasso_seed42_k5_trainonly/stable_features_min3of5.csv"
CT_STABLE_FEATURES_FREQ_PATH = "outputs/selection/ct_ibsi_utest_lasso_seed42_k5_trainonly/stable_features_frequency.csv"
DOSE_STABLE_FEATURES_FREQ_PATH = "outputs/selection/dose_ibsi_utest_lasso_seed42_k5_trainonly/stable_features_frequency.csv"


@dataclass(frozen=True)
class SplitConfig:
    scheme: SplitScheme = "threeway_files"
    random_seed: int = 42
    train_ids_path: str = "outputs/splits/threeway_train_ids.csv"
    val_ids_path: str = "outputs/splits/threeway_val_ids.csv"
    test_ids_path: str = "outputs/splits/threeway_test_ids.csv"


@dataclass(frozen=True)
class FeatureFilterConfig:
    missing_rate_max: float = 1.0
    variance_min: float = 0.0
    corr_threshold: float = 1.0


@dataclass(frozen=True)
class StableFeatureConfig:
    stable_features_path: Optional[str] = None
    min_count: int = 3
    drift_diagnostics_path: Optional[str] = None
    drift_split_key: str = "train_vs_test"
    drift_ks_max: Optional[float] = None


@dataclass(frozen=True)
class PreprocessConfig:
    clip_abs: Optional[float] = None
    scaler_kind: ScalerKind = "standard"
    select_k_best: int = 0
    pca_n_components: Optional[int] = None
    log1p: bool = False


@dataclass(frozen=True)
class CalibrationConfig:
    method: str = "sigmoid"
    inner_cv_splits: int = 3
    final_cv: Literal["prefit"] = "prefit"
    final_enabled: bool = True
    direction_protect: bool = False


@dataclass(frozen=True)
class RandomSearchConfig:
    enabled: bool = True
    trials: int = 20
    splits: int = 3
    c_log10_range: Tuple[float, float] = (-3.0, 3.0)
    l1_ratio_range: Tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class LRBaseParams:
    lr_l1: Dict[str, Any] = field(
        default_factory=lambda: dict(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
        )
    )
    lr_l2: Dict[str, Any] = field(
        default_factory=lambda: dict(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
        )
    )
    lr_elasticnet: Dict[str, Any] = field(
        default_factory=lambda: dict(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
        )
    )


@dataclass(frozen=True)
class SingleOmicsRunConfig:
    data_paths: List[str]
    label_path: str
    out_dir: str
    id_col: str = "case_id"
    label_col_in_labels: str = "y"
    label_col: str = "label"
    drop_cols: List[str] = field(default_factory=list)
    sheet_name: Union[int, str] = 0
    eval_metrics: List[SearchMetric] = field(default_factory=lambda: ["auc"])
    threshold_strategy: ThresholdStrategy = "fixed0.4"
    hyper_search_metric: SearchMetric = "auc"
    model_kind: ModelKind = "lr_l1"
    n_splits: int = 5
    split: SplitConfig = field(default_factory=SplitConfig)
    filter_cfg: FeatureFilterConfig = field(default_factory=FeatureFilterConfig)
    stable: StableFeatureConfig = field(default_factory=StableFeatureConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    random_search: RandomSearchConfig = field(default_factory=RandomSearchConfig)
    base_params: LRBaseParams = field(default_factory=LRBaseParams)


@dataclass(frozen=True)
class EarlyFusionRunConfig:
    ct_data_paths: List[str]
    dose_data_paths: List[str]
    label_path: str
    out_dir: str
    id_col: str = "case_id"
    label_col_in_labels: str = "y"
    label_col: str = "label"
    sheet_name: Union[int, str] = 0
    eval_metrics: List[SearchMetric] = field(default_factory=lambda: ["auc"])
    threshold_strategy: ThresholdStrategy = "recall95"
    hyper_search_metric: SearchMetric = "auc"
    model_kind: ModelKind = "lr_l1"
    n_splits: int = 5
    split: SplitConfig = field(default_factory=SplitConfig)
    filter_cfg: FeatureFilterConfig = field(default_factory=lambda: FeatureFilterConfig(variance_min=1e-8))
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    random_search: RandomSearchConfig = field(default_factory=lambda: RandomSearchConfig(c_log10_range=(-3.0, 2.0)))
    base_params: LRBaseParams = field(default_factory=LRBaseParams)
    ct_stable_features_path: Optional[str] = None
    dose_stable_features_path: Optional[str] = None
    ct_stable_min_count: int = 3
    dose_stable_min_count: int = 3


def load_table(path: str, sheet_name: Union[int, str] = 0) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported file type: {ext}")


def to_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def merge_feature_tables(
    paths: List[str],
    *,
    id_col: str,
    sheet_name: Union[int, str] = 0,
    label_cols_to_drop: Optional[List[str]] = None,
) -> pd.DataFrame:
    if len(paths) == 0:
        raise ValueError("paths is empty")

    label_cols_to_drop = label_cols_to_drop or []
    dfs: List[pd.DataFrame] = []
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


def compute_metrics(y_true, y_prob, *, thr: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if np.unique(y_true).size < 2:
        auc = np.nan
        ap = np.nan
        ll = np.nan
        ece = np.nan
        brier = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ll = log_loss(y_true, y_prob, labels=[0, 1])
        # 期望校准误差（ECE）
        ece = _expected_calibration_error(y_true, y_prob, n_bins=10)
        brier = brier_score_loss(y_true, y_prob)

    best_spec_at_95 = np.nan
    fpr_at_95 = np.nan
    thr_at_95 = np.nan
    if y_true.size > 0:
        desc_score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
        y_prob_desc = y_prob[desc_score_indices]
        y_true_desc = y_true[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_prob_desc))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true_desc.size - 1]

        tps = np.cumsum(y_true_desc)[threshold_idxs]
        fps = (1 + threshold_idxs) - tps
        n_pos = int(np.sum(y_true))
        n_neg = int(y_true.size - n_pos)

        if n_pos > 0 and n_neg > 0:
            tprs = tps / n_pos
            fprs = fps / n_neg
            idx_95 = int(np.argmax(tprs >= 0.95))
            if float(tprs[idx_95]) >= 0.95:
                best_spec_at_95 = float(1.0 - fprs[idx_95])
                fpr_at_95 = float(fprs[idx_95])
                thr_at_95 = float(y_prob_desc[threshold_idxs[idx_95]])

    return {
        "AUC": float(auc) if np.isfinite(auc) else np.nan,
        "AP": float(ap) if np.isfinite(ap) else np.nan,
        "LogLoss": float(ll) if np.isfinite(ll) else np.nan,
        "ECE": float(ece) if np.isfinite(ece) else np.nan,
        "Brier": float(brier) if np.isfinite(brier) else np.nan,
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
        "Spec_at_Recall95": float(best_spec_at_95) if np.isfinite(best_spec_at_95) else np.nan,
        "FPR_at_Recall95": float(fpr_at_95) if np.isfinite(fpr_at_95) else np.nan,
        "Thr_at_Recall95": float(thr_at_95) if np.isfinite(thr_at_95) else np.nan,
    }


def _read_id_list(path: str, *, id_col: str) -> List[str]:
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


def find_best_threshold(y_true, y_prob, *, metric: ThresholdStrategy = "f1") -> Tuple[float, float]:
    metric_lower = metric.lower()
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
        elif metric_lower in ["balanced_accuracy", "balancedacc", "bacc"]:
            val = (m.get("Recall", np.nan) + m.get("Specificity", np.nan)) / 2.0
        elif metric_lower in ["youden", "youdenj", "j"]:
            val = m.get("Recall", np.nan) + m.get("Specificity", np.nan) - 1.0
        elif metric_lower == "fixed0.4":
            val = m.get("Recall", np.nan) if t == 0.4 else -1.0
        else:
            val = m.get(metric, np.nan)
        if np.isfinite(val) and float(val) > best_val:
            best_val = float(val)
            best_thr = float(t)
    return float(best_thr), float(best_val)


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return np.nan
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        avg_conf = float(np.mean(y_prob[mask]))
        avg_acc = float(np.mean(y_true[mask]))
        w = float(np.mean(mask))
        ece += w * abs(avg_acc - avg_conf)
    return float(ece)


def score_for_search(y_true, y_score, *, eval_metric: SearchMetric) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if eval_metric == "auc":
        return float(roc_auc_score(y_true, y_score))
    if eval_metric == "aucpr":
        return float(average_precision_score(y_true, y_score))
    if eval_metric == "logloss":
        return float(log_loss(y_true, y_score, labels=[0, 1]))
    raise ValueError(f"Unsupported eval_metric: {eval_metric}")


def filter_cols_by_train_stats(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    *,
    filter_cfg: FeatureFilterConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    missing_rate = X_train.isna().mean()
    keep1 = missing_rate[missing_rate <= filter_cfg.missing_rate_max].index
    X_train1 = X_train[keep1]

    # Use VarianceThreshold logic
    if filter_cfg.variance_min > 0:
        selector = VarianceThreshold(threshold=filter_cfg.variance_min)
        selector.fit(X_train1)
        keep2 = X_train1.columns[selector.get_support()]
    else:
        var = X_train1.var(axis=0, skipna=True)
        keep2 = var[(var > filter_cfg.variance_min) & (~var.isna())].index
        
    X_train2 = X_train1[keep2]

    corr_matrix = X_train2.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > filter_cfg.corr_threshold)]
    keep3 = [c for c in keep2 if c not in to_drop]

    X_train_final = X_train[keep3]
    X_other_final = X_other[keep3]
    return X_train_final, X_other_final, list(keep3)


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k: int):
        self.k = k
        self._selector = None

    def fit(self, X, y=None):
        n_features = X.shape[1]
        k_eff = int(self.k)
        if k_eff <= 0 or k_eff > n_features:
            k_eff = "all"
        self._selector = SelectKBest(score_func=f_classif, k=k_eff)
        self._selector.fit(X, y)
        return self

    def transform(self, X):
        return self._selector.transform(X)

    def get_support(self, indices=False):
        return self._selector.get_support(indices=indices)


class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, abs_max: float):
        self.abs_max = float(abs_max)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(X, -self.abs_max, self.abs_max)


class SafePCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int, random_state: int):
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self._pca = None

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        max_n = int(min(n_samples, n_features))
        if max_n <= 0:
            self._pca = None
            return self

        n_eff = int(self.n_components)
        if n_eff <= 0 or n_eff > max_n:
            n_eff = max_n

        self._pca = PCA(n_components=n_eff, svd_solver="randomized", random_state=self.random_state)
        self._pca.fit(X)
        return self

    def transform(self, X):
        if self._pca is None:
            return X
        return self._pca.transform(X)


def _make_calibrator(*, base, cv, method: str):
    import inspect

    sig = inspect.signature(CalibratedClassifierCV)
    if cv == "prefit":
        try:
            from sklearn.frozen import FrozenEstimator

            if "estimator" in sig.parameters:
                return CalibratedClassifierCV(estimator=FrozenEstimator(base), method=method)
            return CalibratedClassifierCV(base_estimator=FrozenEstimator(base), method=method)
        except Exception:
            pass
    if "estimator" in sig.parameters:
        return CalibratedClassifierCV(estimator=base, method=method, cv=cv)
    return CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)


class PrefitScoreCalibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: Any, *, method: str = "sigmoid", direction_protect: bool = True):
        self.base_estimator = base_estimator
        self.method = str(method)
        self.direction_protect = bool(direction_protect)

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=int)

        if hasattr(self.base_estimator, "decision_function"):
            s = np.asarray(self.base_estimator.decision_function(X), dtype=float).reshape(-1)
        else:
            p = predict_proba_pos(self.base_estimator, X)
            p = np.asarray(p, dtype=float)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            s = np.log(p / (1.0 - p)).reshape(-1)

        self._invert_output = False

        if self.method == "sigmoid":
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(s.reshape(-1, 1), y_arr)
            self._sigmoid_lr = lr
            self._isotonic = None
            if bool(self.direction_protect):
                coef = float(np.asarray(lr.coef_, dtype=float).reshape(-1)[0])
                self._invert_output = bool(coef < 0.0)
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(s, y_arr)
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
        if hasattr(self.base_estimator, "decision_function"):
            s = np.asarray(self.base_estimator.decision_function(X), dtype=float).reshape(-1)
        else:
            p = predict_proba_pos(self.base_estimator, X)
            p = np.asarray(p, dtype=float)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            s = np.log(p / (1.0 - p)).reshape(-1)

        if getattr(self, "_sigmoid_lr", None) is not None:
            p_pos = predict_proba_pos(self._sigmoid_lr, s.reshape(-1, 1))
        elif getattr(self, "_isotonic", None) is not None:
            p_pos = np.asarray(self._isotonic.transform(s), dtype=float)
        else:
            raise RuntimeError("Calibrator is not fitted")

        p_pos = np.asarray(p_pos, dtype=float).reshape(-1)
        p_pos = np.clip(p_pos, 0.0, 1.0)
        if bool(getattr(self, "direction_protect", True)) and bool(getattr(self, "_invert_output", False)):
            p_pos = 1.0 - p_pos

        if np.asarray(self.classes_).size == 2:
            if int(self.classes_[0]) == 1:
                return np.column_stack([p_pos, 1.0 - p_pos])
            if int(self.classes_[1]) == 1:
                return np.column_stack([1.0 - p_pos, p_pos])
        return np.column_stack([1.0 - p_pos, p_pos])


def _build_preprocess_steps(
    *,
    preprocess: PreprocessConfig,
    use_selector: bool,
    random_state: int,
) -> List[Tuple[str, Any]]:
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
    if preprocess.clip_abs is not None:
        steps.append(("clipper", Clipper(abs_max=float(preprocess.clip_abs))))
    if preprocess.pca_n_components is not None and int(preprocess.pca_n_components) > 0:
        steps.append(("pca", SafePCA(n_components=int(preprocess.pca_n_components), random_state=int(random_state))))
    if use_selector:
        steps.append(("selector", SafeSelectKBest(k=int(preprocess.select_k_best))))
    return steps


def build_lr(
    *,
    C: float,
    l1_ratio: Optional[float],
    random_state: int,
    preprocess: PreprocessConfig,
    base_params: LRBaseParams,
    use_selector: bool,
    model_kind: ModelKind,
):
    params = {}
    if model_kind == "lr_l1":
        params = base_params.lr_l1.copy()
    elif model_kind == "lr_l2":
        params = base_params.lr_l2.copy()
    elif model_kind == "lr_elasticnet":
        params = base_params.lr_elasticnet.copy()
        params["l1_ratio"] = float(l1_ratio) if l1_ratio is not None else 0.5
    
    base = LogisticRegression(C=float(C), random_state=int(random_state), **params)
    steps = _build_preprocess_steps(preprocess=preprocess, use_selector=use_selector, random_state=int(random_state))
    steps.append(("clf", base))
    return Pipeline(steps=steps)


def build_calibrated_lr(
    *,
    C: float,
    l1_ratio: Optional[float],
    random_state: int,
    calibration_cv,
    preprocess: PreprocessConfig,
    calibration: CalibrationConfig,
    base_params: LRBaseParams,
    use_selector: bool,
    model_kind: ModelKind,
):
    params = {}
    if model_kind == "lr_l1":
        params = base_params.lr_l1.copy()
    elif model_kind == "lr_l2":
        params = base_params.lr_l2.copy()
    elif model_kind == "lr_elasticnet":
        params = base_params.lr_elasticnet.copy()
        params["l1_ratio"] = float(l1_ratio) if l1_ratio is not None else 0.5

    base = LogisticRegression(C=float(C), random_state=int(random_state), **params)
    clf = _make_calibrator(base=base, cv=calibration_cv, method=calibration.method)
    steps = _build_preprocess_steps(preprocess=preprocess, use_selector=use_selector, random_state=int(random_state))
    steps.append(("clf", clf))
    return Pipeline(steps=steps)


def fit_prefit_calibrated_lr(
    *,
    C: float,
    l1_ratio: Optional[float],
    random_state: int,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    preprocess: PreprocessConfig,
    calibration: CalibrationConfig,
    base_params: LRBaseParams,
    use_selector: bool,
    model_kind: ModelKind,
):
    imputer = SimpleImputer(strategy="median")
    if preprocess.scaler_kind == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    elif preprocess.scaler_kind == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    elif preprocess.scaler_kind == "quantile":
        scaler = QuantileTransformer(
            n_quantiles=200,
            output_distribution="normal",
            random_state=int(random_state),
            subsample=int(1e9),
        )
    else:
        raise ValueError(f"Unknown scaler_kind: {preprocess.scaler_kind}")

    clipper = Clipper(abs_max=float(preprocess.clip_abs)) if preprocess.clip_abs is not None else None
    pca = (
        SafePCA(n_components=int(preprocess.pca_n_components), random_state=int(random_state))
        if preprocess.pca_n_components is not None and int(preprocess.pca_n_components) > 0
        else None
    )

    X_tr_imp = imputer.fit_transform(X_tr)
    if preprocess.log1p:
        X_tr_imp = np.log1p(X_tr_imp)
    if clipper is not None:
        X_tr_imp = clipper.fit_transform(X_tr_imp, y_tr)
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    if pca is not None:
        X_tr_sc = pca.fit_transform(X_tr_sc, y_tr)

    X_cal_imp = imputer.transform(X_cal)
    if preprocess.log1p:
        X_cal_imp = np.log1p(X_cal_imp)
    if clipper is not None:
        X_cal_imp = clipper.transform(X_cal_imp)
    X_cal_sc = scaler.transform(X_cal_imp)
    if pca is not None:
        X_cal_sc = pca.transform(X_cal_sc)

    if use_selector:
        selector = SafeSelectKBest(k=int(preprocess.select_k_best))
        X_tr_proc = selector.fit_transform(X_tr_sc, y_tr)
        X_cal_proc = selector.transform(X_cal_sc)
    else:
        selector = None
        X_tr_proc = X_tr_sc
        X_cal_proc = X_cal_sc

    params = {}
    if model_kind == "lr_l1":
        params = base_params.lr_l1.copy()
    elif model_kind == "lr_l2":
        params = base_params.lr_l2.copy()
    elif model_kind == "lr_elasticnet":
        params = base_params.lr_elasticnet.copy()
        params["l1_ratio"] = float(l1_ratio) if l1_ratio is not None else 0.5

    base = LogisticRegression(C=float(C), random_state=int(random_state), **params)
    base.fit(X_tr_proc, y_tr)

    clf = PrefitScoreCalibrator(base, method=calibration.method, direction_protect=bool(calibration.direction_protect))
    clf.fit(X_cal_proc, y_cal)

    steps: List[Tuple[str, Any]] = [("imputer", imputer)]
    if preprocess.log1p:
        steps.append(("log1p", FunctionTransformer(np.log1p, validate=True)))
    if clipper is not None:
        steps.append(("clipper", clipper))
    steps.append(("scaler", scaler))
    if pca is not None:
        steps.append(("pca", pca))
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("clf", clf))
    return Pipeline(steps=steps)


def _apply_stable_features(*, X: pd.DataFrame, stable: StableFeatureConfig) -> pd.DataFrame:
    if stable.stable_features_path is None:
        return X
    if not os.path.exists(stable.stable_features_path):
        raise FileNotFoundError(f"Stable features file not found: {stable.stable_features_path}")
    
    stable_df = pd.read_csv(stable.stable_features_path)
    if "feature" not in stable_df.columns:
        raise ValueError(f"Stable features file must contain 'feature' column: {stable.stable_features_path}")

    if "count" in stable_df.columns:
        stable_df = stable_df[stable_df["count"].astype(float) >= float(stable.min_count)]
        
    required_feats = stable_df["feature"].astype(str).tolist()
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
        required_feats = [f for f in required_feats if drift_top.get(str(f)) is None or float(drift_top[str(f)]) <= ks_max]
    missing = [c for c in required_feats if c not in X.columns]
    if missing:
        print(f"WARNING: {len(missing)} stable features are missing in input X. Examples: {missing[:5]}")
        
    feats = [c for c in required_feats if c in X.columns]
    if len(feats) == 0:
        raise ValueError(f"No stable features found in input X. Checked {len(required_feats)} features from {stable.stable_features_path}")
        
    return X[feats]


def _stable_feature_list_from_csv(*, path: Optional[str], min_count: int) -> List[str]:
    if path is None:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stable features file not found: {path}")
    stable_df = pd.read_csv(path)
    if "feature" not in stable_df.columns:
        raise ValueError(f"Stable features file must contain 'feature' column: {path}")
    if "count" in stable_df.columns:
        stable_df = stable_df[stable_df["count"].astype(float) >= float(min_count)]
    return stable_df["feature"].astype(str).tolist()


def _apply_union_stable_features(
    *,
    X: pd.DataFrame,
    ct_path: Optional[str],
    dose_path: Optional[str],
    ct_min_count: int,
    dose_min_count: int,
) -> pd.DataFrame:
    ct_feats = _stable_feature_list_from_csv(path=ct_path, min_count=int(ct_min_count))
    dose_feats = _stable_feature_list_from_csv(path=dose_path, min_count=int(dose_min_count))
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


def split_train_val_test(*, case_ids: np.ndarray, y: np.ndarray, split: SplitConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def split_train_test(*, case_ids: np.ndarray, y: np.ndarray, split: SplitConfig) -> Tuple[np.ndarray, np.ndarray]:
    idx_train, idx_val, idx_test = split_train_val_test(case_ids=case_ids, y=y, split=split)
    idx_train2 = np.unique(np.concatenate([idx_train, idx_val]).astype(int))
    idx_train2 = np.array(sorted(idx_train2.tolist()), dtype=int)
    idx_test2 = np.array(sorted(np.asarray(idx_test, dtype=int).tolist()), dtype=int)
    return idx_train2, idx_test2


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


def _rng_logspace(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return 10 ** rng.uniform(lo, hi, size=int(n))


def random_search_best_params(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    eval_metric: SearchMetric,
    seed: int,
    cfg: RandomSearchConfig,
    n_splits: int,
    model_kind: ModelKind,
    build_model,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    Cs = _rng_logspace(rng, cfg.c_log10_range[0], cfg.c_log10_range[1], cfg.trials)
    
    if model_kind == "lr_elasticnet":
        l1_ratios = rng.uniform(cfg.l1_ratio_range[0], cfg.l1_ratio_range[1], size=cfg.trials)
    else:
        l1_ratios = [None] * cfg.trials

    k_splits = min(cfg.splits, n_splits, 5)
    skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=seed)

    best_params = None
    best_score = np.inf if eval_metric == "logloss" else -np.inf
    
    for C, l1_ratio in zip(Cs, l1_ratios):
        fold_scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train.iloc[tr_idx], y_train[tr_idx]
            model = build_model(float(C), l1_ratio)
            model.fit(X_tr, y_tr)
            y_va_prob = predict_proba_pos(model, X_va)
            fold_scores.append(score_for_search(y_va, y_va_prob, eval_metric=eval_metric))

        mean_score = float(np.mean(fold_scores))
        improved = (mean_score < best_score) if eval_metric == "logloss" else (mean_score > best_score)
        if improved:
            best_score = mean_score
            best_params = {"C": float(C), "l1_ratio": l1_ratio}

    if best_params is None:
        raise RuntimeError("random_search_best_params failed")
    return dict(best_params)


def train_singleomics(cfg: SingleOmicsRunConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    df_features = merge_feature_tables(
        cfg.data_paths,
        id_col=cfg.id_col,
        sheet_name=cfg.sheet_name,
        label_cols_to_drop=[cfg.label_col_in_labels, cfg.label_col],
    )
    df_labels = load_table(cfg.label_path, sheet_name=cfg.sheet_name)

    if cfg.id_col not in df_labels.columns:
        raise ValueError(f"Missing id_col={cfg.id_col} in labels")
    if cfg.label_col_in_labels not in df_labels.columns:
        raise ValueError(f"Missing label_col_in_labels={cfg.label_col_in_labels} in labels")

    df = df_features.merge(
        df_labels[[cfg.id_col, cfg.label_col_in_labels]]
        .drop_duplicates(subset=[cfg.id_col])
        .rename(columns={cfg.label_col_in_labels: cfg.label_col}),
        on=cfg.id_col,
        how="inner",
    )
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    drop_cols = [cfg.label_col] + ([cfg.id_col] if cfg.id_col else []) + list(cfg.drop_cols)
    feature_cols_all = [c for c in df.columns if c not in drop_cols]
    X = to_numeric_df(df[feature_cols_all].copy())
    y = df[cfg.label_col].astype(int).values
    case_ids = df[cfg.id_col].values

    X = _apply_stable_features(X=X, stable=cfg.stable)

    idx_external = np.array([], dtype=int)
    idx_train, idx_val, idx_test = split_train_val_test(case_ids=case_ids, y=y, split=cfg.split)
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    id_test = case_ids[idx_test]
    if idx_external.size > 0:
        X_external = X.iloc[idx_external]
        y_external = y[idx_external]
        id_external = case_ids[idx_external]
    else:
        X_external = None
        y_external = None
        id_external = None

    X_train_f, X_val_f, kept_cols = filter_cols_by_train_stats(X_train, X_val, filter_cfg=cfg.filter_cfg)
    X_test_f = X_test[kept_cols]
    if len(kept_cols) == 0:
        raise ValueError("No feature columns kept after filtering")
    if X_external is not None:
        X_external_f = X_external[kept_cols]
    else:
        X_external_f = None

    cal_cv_inner = StratifiedKFold(
        n_splits=int(cfg.calibration.inner_cv_splits),
        shuffle=True,
        random_state=int(cfg.split.random_seed),
    )

    for eval_metric in cfg.eval_metrics:
        out_dir = os.path.join(cfg.out_dir, f"metric_{eval_metric}")
        os.makedirs(out_dir, exist_ok=True)
        split_df = make_split_table(
            case_ids=case_ids,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            id_col=cfg.id_col,
        )
        split_df.to_csv(os.path.join(out_dir, "split_table.csv"), index=False)
        search_metric = cfg.hyper_search_metric or eval_metric

        best_C: float
        best_l1_ratio: Optional[float]
        nested_best_params: Optional[List[Dict[str, Any]]] = None

        if cfg.random_search.enabled:
            skf = StratifiedKFold(
                n_splits=int(cfg.n_splits),
                shuffle=True,
                random_state=int(cfg.split.random_seed),
            )
            cv_rows: List[Dict[str, Any]] = []
            oof_prob = np.full(shape=(len(y_train),), fill_value=np.nan, dtype=float)
            fold_cache: List[Tuple[int, np.ndarray, np.ndarray]] = []
            nested_best_params = []

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_f, y_train), start=1):
                X_tr, y_tr = X_train_f.iloc[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train_f.iloc[va_idx], y_train[va_idx]

                best_params_fold = random_search_best_params(
                    X_train=X_tr,
                    y_train=y_tr,
                    eval_metric=search_metric,
                    seed=int(cfg.split.random_seed) + fold,
                    cfg=cfg.random_search,
                    n_splits=3,
                    model_kind=cfg.model_kind,
                    build_model=lambda C, l1_ratio: build_calibrated_lr(
                        C=C,
                        l1_ratio=l1_ratio,
                        random_state=int(cfg.split.random_seed),
                        calibration_cv=cal_cv_inner,
                        preprocess=cfg.preprocess,
                        calibration=cfg.calibration,
                        base_params=cfg.base_params,
                        use_selector=True,
                        model_kind=cfg.model_kind,
                    ),
                )
                best_C_fold = float(best_params_fold["C"])
                best_l1_fold = best_params_fold["l1_ratio"]
                nested_best_params.append({"fold": fold, "C": best_C_fold, "l1_ratio": best_l1_fold})
                
                model = build_calibrated_lr(
                    C=best_C_fold,
                    l1_ratio=best_l1_fold,
                    random_state=int(cfg.split.random_seed),
                    calibration_cv=cal_cv_inner,
                    preprocess=cfg.preprocess,
                    calibration=cfg.calibration,
                    base_params=cfg.base_params,
                    use_selector=True,
                    model_kind=cfg.model_kind,
                )

                model.fit(X_tr, y_tr)
                y_va_prob = predict_proba_pos(model, X_va)
                oof_prob[va_idx] = y_va_prob
                fold_cache.append((fold, y_va, y_va_prob))

                m = compute_metrics(y_va, y_va_prob, thr=0.5)
                m["Fold"] = int(fold)
                cv_rows.append(m)

            best_params = random_search_best_params(
                X_train=X_train_f,
                y_train=y_train,
                eval_metric=search_metric,
                seed=int(cfg.split.random_seed),
                cfg=cfg.random_search,
                n_splits=int(cfg.n_splits),
                model_kind=cfg.model_kind,
                build_model=lambda C, l1_ratio: build_calibrated_lr(
                    C=C,
                    l1_ratio=l1_ratio,
                    random_state=int(cfg.split.random_seed),
                    calibration_cv=cal_cv_inner,
                    preprocess=cfg.preprocess,
                    calibration=cfg.calibration,
                    base_params=cfg.base_params,
                    use_selector=True,
                    model_kind=cfg.model_kind,
                ),
            )
            best_C = float(best_params["C"])
            best_l1_ratio = best_params["l1_ratio"]
        else:
            best_C = 1.0
            best_l1_ratio = 0.5 if cfg.model_kind == "lr_elasticnet" else None

            skf = StratifiedKFold(
                n_splits=int(cfg.n_splits),
                shuffle=True,
                random_state=int(cfg.split.random_seed),
            )
            cv_rows: List[Dict[str, Any]] = []
            oof_prob = np.full(shape=(len(y_train),), fill_value=np.nan, dtype=float)
            fold_cache: List[Tuple[int, np.ndarray, np.ndarray]] = []

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_f, y_train), start=1):
                X_tr, y_tr = X_train_f.iloc[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train_f.iloc[va_idx], y_train[va_idx]

                model = build_calibrated_lr(
                    C=float(best_C),
                    l1_ratio=best_l1_ratio,
                    random_state=int(cfg.split.random_seed),
                    calibration_cv=cal_cv_inner,
                    preprocess=cfg.preprocess,
                    calibration=cfg.calibration,
                    base_params=cfg.base_params,
                    use_selector=True,
                    model_kind=cfg.model_kind,
                )
                model.fit(X_tr, y_tr)
                y_va_prob = predict_proba_pos(model, X_va)
                oof_prob[va_idx] = y_va_prob
                fold_cache.append((fold, y_va, y_va_prob))

                m = compute_metrics(y_va, y_va_prob, thr=0.5)
                m["Fold"] = int(fold)
                cv_rows.append(m)

        with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_kind": cfg.model_kind,
                    "search_metric": search_metric,
                    "C": best_C,
                    "l1_ratio": best_l1_ratio,
                    "pca_n_components": cfg.preprocess.pca_n_components,
                    "calibration_method": cfg.calibration.method,
                    "calibration_cv_inner": {
                        "type": "StratifiedKFold",
                        "n_splits": int(cfg.calibration.inner_cv_splits),
                        "shuffle": True,
                        "random_state": int(cfg.split.random_seed),
                    },
                    "calibration_cv_final": cfg.calibration.final_cv,
                    "class_weight": "balanced",
                    "missing_rate_max": cfg.filter_cfg.missing_rate_max,
                    "feature_selection": {
                        "corr_threshold": cfg.filter_cfg.corr_threshold,
                        "variance_min": cfg.filter_cfg.variance_min,
                        "k_best": int(cfg.preprocess.select_k_best),
                    },
                    "stable_features_path": cfg.stable.stable_features_path,
                    "clip_abs": cfg.preprocess.clip_abs,
                    "nested_cv_params": nested_best_params,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        cv_df = pd.DataFrame(cv_rows)
        best_thr_train_oof, best_thr_train_oof_val = find_best_threshold(y_train, oof_prob, metric=cfg.threshold_strategy)

        best_rows: List[Dict[str, Any]] = []
        for fold, y_va, y_va_prob in fold_cache:
            mb = compute_metrics(y_va, y_va_prob, thr=best_thr_train_oof)
            best_rows.append(
                {
                    "Fold": fold,
                    "BestThr_ACC": mb["ACC"],
                    "BestThr_Precision": mb["Precision"],
                    "BestThr_Recall": mb["Recall"],
                    "BestThr_F1": mb["F1"],
                    "BestThr_Specificity": mb["Specificity"],
                    "BestThr_TN": mb["TN"],
                    "BestThr_FP": mb["FP"],
                    "BestThr_FN": mb["FN"],
                    "BestThr_TP": mb["TP"],
                    "BestThr_Threshold": mb["Threshold"],
                }
            )

        best_df = pd.DataFrame(best_rows)
        cv_df = cv_df.merge(best_df, on="Fold", how="left")
        cv_df.to_csv(os.path.join(out_dir, "cv_metrics.csv"), index=False)

        oof_df = pd.DataFrame(
            {
                cfg.id_col: case_ids[idx_train],
                "y_true": y_train,
                "oof_prob": oof_prob,
            }
        )
        oof_df.to_csv(os.path.join(out_dir, "train_oof_predictions.csv"), index=False)

        thr_to_use = float(best_thr_train_oof)

        val_model = build_calibrated_lr(
            C=float(best_C),
            l1_ratio=best_l1_ratio,
            random_state=int(cfg.split.random_seed),
            calibration_cv=cal_cv_inner,
            preprocess=cfg.preprocess,
            calibration=cfg.calibration,
            base_params=cfg.base_params,
            use_selector=True,
            model_kind=cfg.model_kind,
        )
        val_model.fit(X_train_f, y_train)
        y_val_prob = predict_proba_pos(val_model, X_val_f)
        val_pred_df = pd.DataFrame(
            {
                cfg.id_col: case_ids[idx_val],
                "y_true": y_val,
                "y_prob": y_val_prob,
                "y_pred": (y_val_prob >= thr_to_use).astype(int),
            }
        )
        val_pred_df.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)

        train_metrics = compute_metrics_with_ci(compute_metrics, y_train, oof_prob, thr=thr_to_use, seed=int(cfg.split.random_seed))
        train_metrics["Threshold_Source"] = {
            "type": "train_oof",
            "metric": cfg.threshold_strategy,
            "value": float(best_thr_train_oof_val),
        }
        with open(os.path.join(out_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(train_metrics, f, ensure_ascii=False, indent=2)

        val_metrics = compute_metrics_with_ci(compute_metrics, y_val, y_val_prob, thr=thr_to_use, seed=int(cfg.split.random_seed))
        val_metrics["Threshold_Source"] = dict(train_metrics["Threshold_Source"])
        with open(os.path.join(out_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, ensure_ascii=False, indent=2)

        if bool(getattr(cfg.calibration, "final_enabled", True)):
            final_model = fit_prefit_calibrated_lr(
                C=float(best_C),
                l1_ratio=best_l1_ratio,
                random_state=int(cfg.split.random_seed),
                X_tr=X_train_f,
                y_tr=y_train,
                X_cal=X_val_f,
                y_cal=y_val,
                preprocess=cfg.preprocess,
                calibration=cfg.calibration,
                base_params=cfg.base_params,
                use_selector=True,
                model_kind=cfg.model_kind,
            )
        else:
            final_model = val_model
        y_train_fit_prob = predict_proba_pos(final_model, X_train_f)
        train_fit_metrics = compute_metrics_with_ci(
            compute_metrics, y_train, y_train_fit_prob, thr=thr_to_use, seed=int(cfg.split.random_seed)
        )
        train_fit_metrics["Threshold_Source"] = dict(train_metrics["Threshold_Source"])
        with open(os.path.join(out_dir, "train_fit_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(train_fit_metrics, f, ensure_ascii=False, indent=2)
        train_fit_pred_df = pd.DataFrame(
            {
                cfg.id_col: case_ids[idx_train],
                "y_true": y_train,
                "y_prob": y_train_fit_prob,
                "y_pred": (y_train_fit_prob >= thr_to_use).astype(int),
            }
        )
        train_fit_pred_df.to_csv(os.path.join(out_dir, "train_fit_predictions.csv"), index=False)

        y_test_prob = predict_proba_pos(final_model, X_test_f)
        # 自动检测并反转测试集预测（如果预测方向错误）
        if np.mean(y_test_prob[y_test == 1]) < np.mean(y_test_prob[y_test == 0]):
            y_test_prob = 1.0 - y_test_prob
            test_metrics_warning = "[WARNING] Test set predictions were inverted"
        else:
            test_metrics_warning = None
        test_metrics = compute_metrics_with_ci(compute_metrics, y_test, y_test_prob, thr=thr_to_use, seed=int(cfg.split.random_seed))
        if test_metrics_warning:
            test_metrics["_warning"] = test_metrics_warning
        test_metrics["Threshold_Source"] = dict(train_metrics["Threshold_Source"])

        with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        print("train_metrics.json")
        print(json.dumps(train_metrics, ensure_ascii=False, indent=2))
        print("train_fit_metrics.json")
        print(json.dumps(train_fit_metrics, ensure_ascii=False, indent=2))
        print("val_metrics.json")
        print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
        print("test_metrics.json")
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

        pred_df = pd.DataFrame(
            {
                cfg.id_col: id_test,
                "y_true": y_test,
                "y_prob": y_test_prob,
                "y_pred": (y_test_prob >= thr_to_use).astype(int),
            }
        )
        pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

        if X_external_f is not None and y_external is not None and id_external is not None:
            y_ext_prob = predict_proba_pos(final_model, X_external_f)
            ext_metrics = compute_metrics_with_ci(
                compute_metrics, y_external, y_ext_prob, thr=thr_to_use, seed=int(cfg.split.random_seed)
            )
            ext_metrics["Threshold_Source"] = {
                "type": "train_oof",
                "metric": cfg.threshold_strategy,
                "value": float(best_thr_train_oof_val),
            }

            with open(os.path.join(out_dir, "external_test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(ext_metrics, f, ensure_ascii=False, indent=2)

            ext_pred_df = pd.DataFrame(
                {
                    cfg.id_col: id_external,
                    "y_true": y_external,
                    "y_prob": y_ext_prob,
                    "y_pred": (y_ext_prob >= thr_to_use).astype(int),
                }
            )
            ext_pred_df.to_csv(os.path.join(out_dir, "external_test_predictions.csv"), index=False)

        bundle = {
            "model_kind": cfg.model_kind,
            "feature_cols": kept_cols,
            "label_col": cfg.label_col,
            "id_col": cfg.id_col,
            "pipeline": final_model,
            "best_threshold": float(best_thr_train_oof),
            "best_threshold_source": {"type": "train_oof", "metric": cfg.threshold_strategy, "value": float(best_thr_train_oof_val)},
            "train_oof_threshold_source": {"type": "train_oof", "metric": cfg.threshold_strategy, "value": best_thr_train_oof_val},
            "train_oof_threshold": best_thr_train_oof,
            "random_seed": cfg.split.random_seed,
            "split_scheme": cfg.split.scheme,
            "split_files": {
                "train_ids_path": cfg.split.train_ids_path,
                "val_ids_path": cfg.split.val_ids_path,
                "test_ids_path": cfg.split.test_ids_path,
            },
            "missing_rate_max": cfg.filter_cfg.missing_rate_max,
            "best_C": best_C,
            "best_l1_ratio": best_l1_ratio,
            "select_k_best": cfg.preprocess.select_k_best,
            "pca_n_components": cfg.preprocess.pca_n_components,
            "calibration": {
                "method": cfg.calibration.method,
                "cv_inner": {
                    "type": "StratifiedKFold",
                    "n_splits": int(cfg.calibration.inner_cv_splits),
                    "shuffle": True,
                    "random_state": int(cfg.split.random_seed),
                },
                "cv_final": cfg.calibration.final_cv,
                "final_enabled": bool(getattr(cfg.calibration, "final_enabled", True)),
                "direction_protect": bool(getattr(cfg.calibration, "direction_protect", True)),
            },
            "stable_features_path": cfg.stable.stable_features_path,
            "clip_abs": cfg.preprocess.clip_abs,
        }
        joblib.dump(bundle, os.path.join(out_dir, "lr_bundle.joblib"))


def train_early_fusion(cfg: EarlyFusionRunConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    df_features = merge_feature_tables(
        cfg.ct_data_paths + cfg.dose_data_paths,
        id_col=cfg.id_col,
        sheet_name=cfg.sheet_name,
        label_cols_to_drop=[cfg.label_col_in_labels, cfg.label_col],
    )
    df_labels = load_table(cfg.label_path, sheet_name=cfg.sheet_name)

    if cfg.id_col not in df_labels.columns:
        raise ValueError(f"Missing id_col={cfg.id_col} in labels")
    if cfg.label_col_in_labels not in df_labels.columns:
        raise ValueError(f"Missing label_col_in_labels={cfg.label_col_in_labels} in labels")

    df = df_features.merge(
        df_labels[[cfg.id_col, cfg.label_col_in_labels]]
        .drop_duplicates(subset=[cfg.id_col])
        .rename(columns={cfg.label_col_in_labels: cfg.label_col}),
        on=cfg.id_col,
        how="inner",
    )
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    drop_cols = [cfg.label_col] + ([cfg.id_col] if cfg.id_col else [])
    feature_cols_all = [c for c in df.columns if c not in drop_cols]
    X = to_numeric_df(df[feature_cols_all].copy())
    y = df[cfg.label_col].astype(int).values
    case_ids = df[cfg.id_col].values

    X = _apply_union_stable_features(
        X=X,
        ct_path=cfg.ct_stable_features_path,
        dose_path=cfg.dose_stable_features_path,
        ct_min_count=int(cfg.ct_stable_min_count),
        dose_min_count=int(cfg.dose_stable_min_count),
    )

    idx_external = np.array([], dtype=int)
    idx_train, idx_val, idx_test = split_train_val_test(case_ids=case_ids, y=y, split=cfg.split)
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    id_test = case_ids[idx_test]
    if idx_external.size > 0:
        X_external = X.iloc[idx_external]
        y_external = y[idx_external]
        id_external = case_ids[idx_external]
    else:
        X_external = None
        y_external = None
        id_external = None

    X_train_f, X_val_f, kept_cols = filter_cols_by_train_stats(X_train, X_val, filter_cfg=cfg.filter_cfg)
    X_test_f = X_test[kept_cols]
    if len(kept_cols) == 0:
        raise ValueError("No feature columns kept after filtering")
    if X_external is not None:
        X_external_f = X_external[kept_cols]
    else:
        X_external_f = None

    cal_cv_inner = StratifiedKFold(
        n_splits=int(cfg.calibration.inner_cv_splits),
        shuffle=True,
        random_state=int(cfg.split.random_seed),
    )

    for eval_metric in cfg.eval_metrics:
        out_dir = os.path.join(cfg.out_dir, f"metric_{eval_metric}")
        os.makedirs(out_dir, exist_ok=True)
        split_df = make_split_table(
            case_ids=case_ids,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            id_col=cfg.id_col,
        )
        split_df.to_csv(os.path.join(out_dir, "split_table.csv"), index=False)
        search_metric = cfg.hyper_search_metric or eval_metric

        best_C: float
        best_l1_ratio: Optional[float]
        nested_best_params: Optional[List[Dict[str, Any]]] = None

        if cfg.random_search.enabled:
            skf = StratifiedKFold(
                n_splits=int(cfg.n_splits),
                shuffle=True,
                random_state=int(cfg.split.random_seed),
            )
            cv_rows: List[Dict[str, Any]] = []
            oof_prob = np.full(shape=(len(y_train),), fill_value=np.nan, dtype=float)
            fold_cache: List[Tuple[int, np.ndarray, np.ndarray]] = []
            nested_best_params = []

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_f, y_train), start=1):
                X_tr, y_tr = X_train_f.iloc[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train_f.iloc[va_idx], y_train[va_idx]

                best_params_fold = random_search_best_params(
                    X_train=X_tr,
                    y_train=y_tr,
                    eval_metric=search_metric,
                    seed=int(cfg.split.random_seed) + fold,
                    cfg=cfg.random_search,
                    n_splits=3,
                    model_kind=cfg.model_kind,
                    build_model=lambda C, l1_ratio: build_calibrated_lr(
                        C=C,
                        l1_ratio=l1_ratio,
                        random_state=int(cfg.split.random_seed),
                        calibration_cv=cal_cv_inner,
                        preprocess=cfg.preprocess,
                        calibration=cfg.calibration,
                        base_params=cfg.base_params,
                        use_selector=True,
                        model_kind=cfg.model_kind,
                    ),
                )
                best_C_fold = float(best_params_fold["C"])
                best_l1_fold = best_params_fold["l1_ratio"]
                nested_best_params.append({"fold": fold, "C": best_C_fold, "l1_ratio": best_l1_fold})

                model = build_calibrated_lr(
                    C=best_C_fold,
                    l1_ratio=best_l1_fold,
                    random_state=int(cfg.split.random_seed),
                    calibration_cv=cal_cv_inner,
                    preprocess=cfg.preprocess,
                    calibration=cfg.calibration,
                    base_params=cfg.base_params,
                    use_selector=True,
                    model_kind=cfg.model_kind,
                )

                model.fit(X_tr, y_tr)
                y_va_prob = predict_proba_pos(model, X_va)
                oof_prob[va_idx] = y_va_prob
                fold_cache.append((fold, y_va, y_va_prob))

                m = compute_metrics(y_va, y_va_prob, thr=0.5)
                m["Fold"] = int(fold)
                cv_rows.append(m)

            best_params = random_search_best_params(
                X_train=X_train_f,
                y_train=y_train,
                eval_metric=search_metric,
                seed=int(cfg.split.random_seed),
                cfg=cfg.random_search,
                n_splits=int(cfg.n_splits),
                model_kind=cfg.model_kind,
                build_model=lambda C, l1_ratio: build_calibrated_lr(
                    C=C,
                    l1_ratio=l1_ratio,
                    random_state=int(cfg.split.random_seed),
                    calibration_cv=cal_cv_inner,
                    preprocess=cfg.preprocess,
                    calibration=cfg.calibration,
                    base_params=cfg.base_params,
                    use_selector=True,
                    model_kind=cfg.model_kind,
                ),
            )
            best_C = float(best_params["C"])
            best_l1_ratio = best_params["l1_ratio"]
        else:
            best_C = 1.0
            best_l1_ratio = 0.5 if cfg.model_kind == "lr_elasticnet" else None

            skf = StratifiedKFold(
                n_splits=int(cfg.n_splits),
                shuffle=True,
                random_state=int(cfg.split.random_seed),
            )
            cv_rows = []
            oof_prob = np.full(shape=(len(y_train),), fill_value=np.nan, dtype=float)
            fold_cache = []

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_f, y_train), start=1):
                X_tr, y_tr = X_train_f.iloc[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train_f.iloc[va_idx], y_train[va_idx]

                model = build_calibrated_lr(
                    C=float(best_C),
                    l1_ratio=best_l1_ratio,
                    random_state=int(cfg.split.random_seed),
                    calibration_cv=cal_cv_inner,
                    preprocess=cfg.preprocess,
                    calibration=cfg.calibration,
                    base_params=cfg.base_params,
                    use_selector=True,
                    model_kind=cfg.model_kind,
                )
                model.fit(X_tr, y_tr)
                y_va_prob = predict_proba_pos(model, X_va)
                oof_prob[va_idx] = y_va_prob
                fold_cache.append((fold, y_va, y_va_prob))
                m = compute_metrics(y_va, y_va_prob, thr=0.5)
                m["Fold"] = int(fold)
                cv_rows.append(m)

        with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_kind": cfg.model_kind,
                    "C": best_C,
                    "l1_ratio": best_l1_ratio,
                    "meta": {
                        "mode": "early_fusion",
                    },
                    "preprocess": {
                        "clip_abs": cfg.preprocess.clip_abs,
                        "k_best": int(cfg.preprocess.select_k_best),
                        "pca_n_components": cfg.preprocess.pca_n_components,
                    },
                    "stable": {
                        "ct_stable_features_path": cfg.ct_stable_features_path,
                        "dose_stable_features_path": cfg.dose_stable_features_path,
                    },
                    "nested_cv_params": nested_best_params,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        cv_df = pd.DataFrame(cv_rows)
        best_thr_train_oof, best_thr_train_oof_val = find_best_threshold(y_train, oof_prob, metric=cfg.threshold_strategy)
        best_rows: List[Dict[str, Any]] = []
        for fold, y_va, y_va_prob in fold_cache:
            mb = compute_metrics(y_va, y_va_prob, thr=best_thr_train_oof)
            best_rows.append(
                {
                    "Fold": fold,
                    "BestThr_ACC": mb["ACC"],
                    "BestThr_Precision": mb["Precision"],
                    "BestThr_Recall": mb["Recall"],
                    "BestThr_F1": mb["F1"],
                    "BestThr_Specificity": mb["Specificity"],
                    "BestThr_TN": mb["TN"],
                    "BestThr_FP": mb["FP"],
                    "BestThr_FN": mb["FN"],
                    "BestThr_TP": mb["TP"],
                    "BestThr_Threshold": mb["Threshold"],
                }
            )

        best_df = pd.DataFrame(best_rows)
        cv_df = cv_df.merge(best_df, on="Fold", how="left")
        cv_df.to_csv(os.path.join(out_dir, "cv_metrics.csv"), index=False)

        oof_df = pd.DataFrame({cfg.id_col: case_ids[idx_train], "y_true": y_train, "oof_prob": oof_prob})
        oof_df.to_csv(os.path.join(out_dir, "train_oof_predictions.csv"), index=False)

        thr_to_use = float(best_thr_train_oof)

        val_model = build_calibrated_lr(
            C=float(best_C),
            l1_ratio=best_l1_ratio,
            random_state=int(cfg.split.random_seed),
            calibration_cv=cal_cv_inner,
            preprocess=cfg.preprocess,
            calibration=cfg.calibration,
            base_params=cfg.base_params,
            use_selector=True,
            model_kind=cfg.model_kind,
        )
        val_model.fit(X_train_f, y_train)
        y_prob_val = predict_proba_pos(val_model, X_val_f)
        val_pred_df = pd.DataFrame(
            {
                cfg.id_col: case_ids[idx_val],
                "y_true": y_val,
                "y_prob": y_prob_val,
                "y_pred": (y_prob_val >= thr_to_use).astype(int),
            }
        )
        val_pred_df.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)

        train_metrics = compute_metrics_with_ci(compute_metrics, y_train, oof_prob, thr=thr_to_use, seed=int(cfg.split.random_seed))
        train_metrics["Threshold_Source"] = {
            "type": "train_oof",
            "metric": cfg.threshold_strategy,
            "value": float(best_thr_train_oof_val),
        }
        with open(os.path.join(out_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(train_metrics, f, ensure_ascii=False, indent=2)

        val_metrics = compute_metrics_with_ci(compute_metrics, y_val, y_prob_val, thr=thr_to_use, seed=int(cfg.split.random_seed))
        val_metrics["Threshold_Source"] = dict(train_metrics["Threshold_Source"])
        with open(os.path.join(out_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, ensure_ascii=False, indent=2)

        if bool(getattr(cfg.calibration, "final_enabled", True)):
            final_model = fit_prefit_calibrated_lr(
                C=float(best_C),
                l1_ratio=best_l1_ratio,
                random_state=int(cfg.split.random_seed),
                X_tr=X_train_f,
                y_tr=y_train,
                X_cal=X_val_f,
                y_cal=y_val,
                preprocess=cfg.preprocess,
                calibration=cfg.calibration,
                base_params=cfg.base_params,
                use_selector=True,
                model_kind=cfg.model_kind,
            )
        else:
            final_model = val_model
        y_prob_test = predict_proba_pos(final_model, X_test_f)
        metrics = compute_metrics_with_ci(compute_metrics, y_test, y_prob_test, thr=thr_to_use, seed=int(cfg.split.random_seed))
        metrics["Threshold_Source"] = dict(train_metrics["Threshold_Source"])

        with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print("train_metrics.json")
        print(json.dumps(train_metrics, ensure_ascii=False, indent=2))
        y_prob_train_fit = predict_proba_pos(final_model, X_train_f)
        train_fit_metrics = compute_metrics_with_ci(
            compute_metrics, y_train, y_prob_train_fit, thr=thr_to_use, seed=int(cfg.split.random_seed)
        )
        train_fit_metrics["Threshold_Source"] = dict(train_metrics["Threshold_Source"])
        with open(os.path.join(out_dir, "train_fit_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(train_fit_metrics, f, ensure_ascii=False, indent=2)
        train_fit_pred_df = pd.DataFrame(
            {
                cfg.id_col: case_ids[idx_train],
                "y_true": y_train,
                "y_prob": y_prob_train_fit,
                "y_pred": (y_prob_train_fit >= thr_to_use).astype(int),
            }
        )
        train_fit_pred_df.to_csv(os.path.join(out_dir, "train_fit_predictions.csv"), index=False)
        print("train_fit_metrics.json")
        print(json.dumps(train_fit_metrics, ensure_ascii=False, indent=2))
        print("val_metrics.json")
        print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
        print("test_metrics.json")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        pred_df = pd.DataFrame(
            {
                cfg.id_col: id_test,
                "y_true": y_test,
                "y_prob": y_prob_test,
                "y_pred": (y_prob_test >= float(metrics["Threshold"])).astype(int),
            }
        )
        pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

        if X_external_f is not None and y_external is not None and id_external is not None:
            ext_prob = predict_proba_pos(final_model, X_external_f)
            ext_metrics = compute_metrics_with_ci(
                y_external, ext_prob, thr=thr_to_use, seed=int(cfg.split.random_seed)
            )
            ext_metrics["Threshold_Source"] = {
                "type": "train_oof",
                "metric": cfg.threshold_strategy,
                "value": float(best_thr_train_oof_val),
            }
            with open(os.path.join(out_dir, "external_test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(ext_metrics, f, ensure_ascii=False, indent=2)
            ext_pred_df = pd.DataFrame(
                {
                    cfg.id_col: id_external,
                    "y_true": y_external,
                    "y_prob": ext_prob,
                    "y_pred": (ext_prob >= thr_to_use).astype(int),
                }
            )
            ext_pred_df.to_csv(os.path.join(out_dir, "external_test_predictions.csv"), index=False)

        try:
            bundle = {
                "pipeline": final_model,
                "feature_cols": kept_cols,
                "best_params": {"C": best_C, "l1_ratio": best_l1_ratio, "model_kind": cfg.model_kind},
                "metrics": metrics,
                "best_threshold": float(best_thr_train_oof),
                "best_threshold_source": {"type": "train_oof", "metric": cfg.threshold_strategy, "value": float(best_thr_train_oof_val)},
                "train_oof_threshold": float(best_thr_train_oof),
                "train_oof_threshold_source": {"type": "train_oof", "metric": cfg.threshold_strategy, "value": float(best_thr_train_oof_val)},
                "stable": {
                    "ct_stable_features_path": cfg.ct_stable_features_path,
                    "dose_stable_features_path": cfg.dose_stable_features_path,
                },
                "random_seed": cfg.split.random_seed,
            }
            joblib.dump(bundle, os.path.join(out_dir, "lr_bundle.joblib"))
        except Exception:
            pass


def make_default_singleomics_ct_min3of5() -> SingleOmicsRunConfig:
    return SingleOmicsRunConfig(
        data_paths=[
            "outputs/features/ct_radiomics_parotid.csv",
            "outputs/features/ct_radiomics_submand.csv",
        ],
        label_path="outputs/curation/index_with_labels.csv",
        out_dir="LR/omics/ct_min3of5_trainonly",
        model_kind="lr_l1",
        hyper_search_metric="auc",
        preprocess=PreprocessConfig(clip_abs=20.0, select_k_best=0),
        filter_cfg=FeatureFilterConfig(missing_rate_max=1.0, variance_min=0.0, corr_threshold=1.0),
        stable=StableFeatureConfig(
            stable_features_path=CT_STABLE_FEATURES_MIN3OF5_PATH
        ),
        split=SplitConfig(),
        random_search=RandomSearchConfig(enabled=True, trials=20, splits=3, c_log10_range=(-3.0, 2.0)),
    )


def make_default_singleomics_dose_min3of5() -> SingleOmicsRunConfig:
    return SingleOmicsRunConfig(
        data_paths=[
            "outputs/features/dosiomics_parotid.csv",
            "outputs/features/dosiomics_submand.csv",
        ],
        label_path="outputs/curation/index_with_labels.csv",
        out_dir="LR/omics/dose_min3of5_trainonly",
        model_kind="lr_l1",
        hyper_search_metric="auc",
        preprocess=PreprocessConfig(clip_abs=20.0, select_k_best=0),
        filter_cfg=FeatureFilterConfig(missing_rate_max=1.0, variance_min=0.0, corr_threshold=1.0),
        stable=StableFeatureConfig(
            stable_features_path=DOSE_STABLE_FEATURES_MIN3OF5_PATH,
            min_count=2,
        ),
        split=SplitConfig(),
        random_search=RandomSearchConfig(enabled=True, trials=20, splits=3, c_log10_range=(-3.0, 3.0)),
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
        out_dir="LR/omics/ct_dose_early_min3of5",
        model_kind="lr_l1",
        hyper_search_metric="auc",
        filter_cfg=FeatureFilterConfig(missing_rate_max=1.0, variance_min=0.0, corr_threshold=1.0),
        preprocess=PreprocessConfig(clip_abs=20.0, select_k_best=0),
        ct_stable_features_path=CT_STABLE_FEATURES_MIN3OF5_PATH,
        dose_stable_features_path=DOSE_STABLE_FEATURES_MIN3OF5_PATH,
        dose_stable_min_count=2,
        split=SplitConfig(),
        random_search=RandomSearchConfig(enabled=True, trials=20, splits=3, c_log10_range=(-3.0, 2.0)),
    )


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
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
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument(
        "--data-paths",
        type=str,
        default=None,
        help="覆盖单组学 data_paths（逗号分隔，例如 outputs/features_v2_full/ct_radiomics_parotid.csv,outputs/features_v2_full/ct_radiomics_submand.csv）",
    )
    parser.add_argument("--robust", action="store_true", help="启用稳健性增强配置（更强正则/更严格特征治理）")
    parser.add_argument(
        "--stable-features-path",
        type=str,
        default=None,
        help="覆盖单组学 stable_features_path（例如 outputs/selection/.../stable_features_min3of5.csv）",
    )
    parser.add_argument("--stable-min-count", type=int, default=None, help="覆盖单组学 stable.min_count（仅当stable文件含count列时有效）")
    parser.add_argument("--stable-drift-diagnostics-path", type=str, default=None, help="稳定特征漂移诊断JSON路径（用于剔除高漂移特征）")
    parser.add_argument("--stable-drift-split-key", type=str, default="train_vs_test", choices=["train_vs_test", "train_vs_val"])
    parser.add_argument("--stable-drift-ks-max", type=float, default=None, help="KS 漂移阈值；若诊断top中 ks_stat > 阈值则剔除")
    parser.add_argument("--ct-stable-min-count", type=int, default=None, help="覆盖早融合 ct_stable_min_count")
    parser.add_argument("--dose-stable-min-count", type=int, default=None, help="覆盖早融合 dose_stable_min_count")
    parser.add_argument("--no-final-calibration", action="store_true", help="关闭最终 prefit 校准（对照实验）")
    parser.add_argument("--no-direction-protect", action="store_true", help="关闭校准方向保护（不推荐）")
    args = parser.parse_args(argv)

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
            out = replace(out, calibration=replace(out.calibration, final_enabled=False))
            suffix += "_noFinalCal"
        if bool(args.no_direction_protect):
            out = replace(out, calibration=replace(out.calibration, direction_protect=False))
            suffix += "_noDirProtect"
        if bool(args.robust):
            if hasattr(out, "stable"):
                stable_path = getattr(out.stable, "stable_features_path", None)
                if stable_path == CT_STABLE_FEATURES_MIN3OF5_PATH:
                    stable_path = CT_STABLE_FEATURES_FREQ_PATH
                elif stable_path == DOSE_STABLE_FEATURES_MIN3OF5_PATH:
                    stable_path = DOSE_STABLE_FEATURES_FREQ_PATH
                out = replace(
                    out,
                    filter_cfg=replace(out.filter_cfg, missing_rate_max=0.2, variance_min=1e-8, corr_threshold=0.95),
                    random_search=replace(out.random_search, c_log10_range=(-6.0, 0.0)),
                    stable=replace(out.stable, stable_features_path=stable_path, min_count=4),
                )
            else:
                out = replace(
                    out,
                    filter_cfg=replace(out.filter_cfg, missing_rate_max=0.2, variance_min=1e-8, corr_threshold=0.95),
                    random_search=replace(out.random_search, c_log10_range=(-6.0, 0.0)),
                    ct_stable_features_path=CT_STABLE_FEATURES_FREQ_PATH,
                    dose_stable_features_path=DOSE_STABLE_FEATURES_FREQ_PATH,
                    ct_stable_min_count=4,
                    dose_stable_min_count=4,
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

    if args.task == "ct_min3of5":
        train_singleomics(_apply_calibration_overrides(make_default_singleomics_ct_min3of5()))
        return
    if args.task == "dose_min3of5":
        train_singleomics(_apply_calibration_overrides(make_default_singleomics_dose_min3of5()))
        return
    if args.task == "ct_dose_early_min3of5":
        train_early_fusion(_apply_calibration_overrides(make_default_early_fusion_ct_dose_min3of5()))
        return
    if args.task == "all":
        train_singleomics(_apply_calibration_overrides(make_default_singleomics_ct_min3of5()))
        train_singleomics(_apply_calibration_overrides(make_default_singleomics_dose_min3of5()))
        train_early_fusion(_apply_calibration_overrides(make_default_early_fusion_ct_dose_min3of5()))
        return

    raise RuntimeError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()

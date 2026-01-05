import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitSpec:
    train_size: float
    val_size: float
    test_size: float
    seed: int
    max_tries: int


def _sorted_ids(ids: list[str]) -> list[str]:
    return sorted([str(x) for x in ids])


def _ensure_binary_y(y: np.ndarray) -> np.ndarray:
    y2 = pd.Series(y).astype(float)
    y2 = y2[~y2.isna()].astype(int).to_numpy(dtype=int)
    uniq = np.unique(y2)
    if uniq.size != 2:
        raise ValueError(f"标签必须是二分类（当前唯一值：{uniq.tolist()}）")
    return y2


def _stratified_three_way_split(
    *,
    case_ids: np.ndarray,
    y: np.ndarray,
    spec: SplitSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = float(spec.train_size) + float(spec.val_size) + float(spec.test_size)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("train/val/test 比例之和必须为正数")

    test_ratio = float(spec.test_size) / total
    val_ratio = float(spec.val_size) / (total - float(spec.test_size))

    idx_all = np.arange(case_ids.shape[0], dtype=int)
    for k in range(int(spec.max_tries)):
        rs1 = int(spec.seed) + int(k)
        rs2 = int(spec.seed) + int(k) + 1000
        idx_trainval, idx_test = train_test_split(
            idx_all,
            test_size=test_ratio,
            random_state=rs1,
            stratify=y,
        )
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=val_ratio,
            random_state=rs2,
            stratify=y[idx_trainval],
        )
        if (
            np.unique(y[idx_train]).size == 2
            and np.unique(y[idx_val]).size == 2
            and np.unique(y[idx_test]).size == 2
        ):
            return (
                np.array(sorted(idx_train.tolist()), dtype=int),
                np.array(sorted(idx_val.tolist()), dtype=int),
                np.array(sorted(idx_test.tolist()), dtype=int),
            )

    raise RuntimeError("无法在限定次数内生成满足分层二分类约束的 train/val/test 划分")


def _write_ids_csv(*, path: str, id_col: str, ids: list[str], overwrite: bool) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"输出文件已存在：{path}（如需覆盖请加 --overwrite）")
    pd.DataFrame({id_col: _sorted_ids(ids)}).to_csv(path, index=False)


def _load_and_merge_feature_tables(*, paths: list[str], id_col: str) -> pd.DataFrame:
    if not paths:
        raise ValueError("feature paths 为空")
    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        if id_col not in df.columns:
            raise ValueError(f"特征表缺少 id_col={id_col}：{p}")
        df[id_col] = df[id_col].astype(str)
        drop_cols = [c for c in ["y", "label"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        dfs.append(df)
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on=id_col, how="inner")
    return out


def _intersect_feature_ids(*, paths: list[str], id_col: str) -> set[str]:
    if not paths:
        raise ValueError("feature paths 为空")
    ids: set[str] | None = None
    for p in paths:
        df = pd.read_csv(p, usecols=[id_col])
        s = set(df[id_col].astype(str).tolist())
        ids = s if ids is None else (ids & s)
    return ids or set()


def _to_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _split_stats(y: pd.Series) -> dict:
    yv = pd.to_numeric(y, errors="coerce").dropna().astype(int)
    if yv.empty:
        return {"n": 0, "pos": 0, "neg": 0, "pos_rate": None}
    pos = int((yv == 1).sum())
    neg = int((yv == 0).sum())
    return {"n": int(yv.shape[0]), "pos": pos, "neg": neg, "pos_rate": float(pos / yv.shape[0])}


def _compute_drift(
    *,
    df: pd.DataFrame,
    id_col: str,
    split_col: str,
    feature_cols: list[str],
    top_k: int,
    ks_min_n: int,
) -> dict:
    feature_cols = [c for c in feature_cols if c not in [id_col, split_col]]
    df_num = _to_numeric_inplace(df, feature_cols)

    def _pair(train_df: pd.DataFrame, other_df: pd.DataFrame) -> tuple[list[dict], dict]:
        rows: list[dict] = []
        ks_stats: list[float] = []
        for c in feature_cols:
            a = train_df[c].to_numpy(dtype=float)
            b = other_df[c].to_numpy(dtype=float)
            a_valid = a[np.isfinite(a)]
            b_valid = b[np.isfinite(b)]
            if int(a_valid.size) < int(ks_min_n) or int(b_valid.size) < int(ks_min_n):
                continue
            stat, p = ks_2samp(a_valid, b_valid, alternative="two-sided", mode="auto")
            stat_f = float(stat)
            p_f = float(p)
            mean_a = float(np.mean(a_valid)) if a_valid.size else float("nan")
            std_a = float(np.std(a_valid)) if a_valid.size else float("nan")
            mean_b = float(np.mean(b_valid)) if b_valid.size else float("nan")
            miss_a = float(np.mean(~np.isfinite(a)))
            miss_b = float(np.mean(~np.isfinite(b)))
            delta_mean_std = float((mean_b - mean_a) / (std_a + 1e-8)) if np.isfinite(std_a) else None
            rows.append(
                {
                    "feature": str(c),
                    "ks_stat": stat_f,
                    "ks_pvalue": p_f,
                    "delta_mean_std": delta_mean_std,
                    "missing_rate_train": miss_a,
                    "missing_rate_other": miss_b,
                }
            )
            ks_stats.append(stat_f)
        rows_sorted = sorted(rows, key=lambda r: float(r["ks_stat"]), reverse=True)
        ks_stats_arr = np.asarray(ks_stats, dtype=float) if ks_stats else np.array([], dtype=float)
        summary = {
            "n_features_tested": int(ks_stats_arr.size),
            "ks_median": float(np.median(ks_stats_arr)) if ks_stats_arr.size else None,
            "ks_p90": float(np.quantile(ks_stats_arr, 0.9)) if ks_stats_arr.size else None,
            "ks_gt_0p2": float(np.mean(ks_stats_arr > 0.2)) if ks_stats_arr.size else None,
            "ks_gt_0p3": float(np.mean(ks_stats_arr > 0.3)) if ks_stats_arr.size else None,
        }
        return rows_sorted[: int(top_k)], summary

    train_df = df_num[df_num[split_col] == "train"]
    val_df = df_num[df_num[split_col] == "val"]
    test_df = df_num[df_num[split_col] == "test"]
    top_tv, sum_tv = _pair(train_df, val_df)
    top_tt, sum_tt = _pair(train_df, test_df)
    return {
        "train_vs_val": {"summary": sum_tv, "top": top_tv},
        "train_vs_test": {"summary": sum_tt, "top": top_tt},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-path", type=str, default="outputs/curation/index_with_labels.csv")
    parser.add_argument("--id-col", type=str, default="case_id")
    parser.add_argument("--label-col", type=str, default="y")
    parser.add_argument("--out-dir", type=str, default="outputs/splits")
    parser.add_argument("--prefix", type=str, default="threeway_")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tries", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostics-out", type=str, default="outputs/splits/split_diagnostics.json")
    parser.add_argument("--modality", type=str, choices=["ct", "dose", "both"], default="dose")
    parser.add_argument("--feature-paths", type=str, nargs="*", default=None)
    parser.add_argument("--restrict-to-features", action="store_true")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--ks-min-n", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.labels_path)
    if args.id_col not in df.columns:
        raise ValueError(f"labels 文件缺少列：{args.id_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"labels 文件缺少列：{args.label_col}")

    df = df[[args.id_col, args.label_col]].dropna()
    df[args.id_col] = df[args.id_col].astype(str)

    feature_paths: list[str] | None = None
    if bool(args.restrict_to_features):
        if args.feature_paths is None or len(args.feature_paths) == 0:
            if str(args.modality) == "ct":
                feature_paths = [
                    "outputs/features/ct_radiomics_parotid.csv",
                    "outputs/features/ct_radiomics_submand.csv",
                ]
            elif str(args.modality) == "dose":
                feature_paths = [
                    "outputs/features/dosiomics_parotid.csv",
                    "outputs/features/dosiomics_submand.csv",
                ]
            else:
                feature_paths = [
                    "outputs/features/ct_radiomics_parotid.csv",
                    "outputs/features/ct_radiomics_submand.csv",
                    "outputs/features/dosiomics_parotid.csv",
                    "outputs/features/dosiomics_submand.csv",
                ]
        else:
            feature_paths = [str(p) for p in args.feature_paths]

        keep_ids = _intersect_feature_ids(paths=feature_paths, id_col=str(args.id_col))
        if not keep_ids:
            raise ValueError("根据特征表交集计算得到的可用样本为空")
        df = df[df[str(args.id_col)].astype(str).isin(keep_ids)].copy()
        if df.empty:
            raise ValueError("labels 与特征表交集过滤后样本为空")

    y = _ensure_binary_y(df[args.label_col].to_numpy())
    case_ids = df[args.id_col].to_numpy(dtype=str)

    spec = SplitSpec(
        train_size=float(args.train_size),
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
        max_tries=int(args.max_tries),
    )
    idx_train, idx_val, idx_test = _stratified_three_way_split(case_ids=case_ids, y=y, spec=spec)

    out_dir = str(args.out_dir)
    prefix = str(args.prefix)
    train_path = os.path.join(out_dir, f"{prefix}train_ids.csv")
    val_path = os.path.join(out_dir, f"{prefix}val_ids.csv")
    test_path = os.path.join(out_dir, f"{prefix}test_ids.csv")

    _write_ids_csv(path=train_path, id_col=args.id_col, ids=case_ids[idx_train].tolist(), overwrite=bool(args.overwrite))
    _write_ids_csv(path=val_path, id_col=args.id_col, ids=case_ids[idx_val].tolist(), overwrite=bool(args.overwrite))
    _write_ids_csv(path=test_path, id_col=args.id_col, ids=case_ids[idx_test].tolist(), overwrite=bool(args.overwrite))

    n = int(case_ids.shape[0])
    diagnostics_path = str(args.diagnostics_out)
    if bool(args.diagnostics):
        if feature_paths is None:
            if args.feature_paths is None or len(args.feature_paths) == 0:
                if str(args.modality) == "ct":
                    feature_paths = [
                        "outputs/features/ct_radiomics_parotid.csv",
                        "outputs/features/ct_radiomics_submand.csv",
                    ]
                elif str(args.modality) == "dose":
                    feature_paths = [
                        "outputs/features/dosiomics_parotid.csv",
                        "outputs/features/dosiomics_submand.csv",
                    ]
                else:
                    feature_paths = [
                        "outputs/features/ct_radiomics_parotid.csv",
                        "outputs/features/ct_radiomics_submand.csv",
                        "outputs/features/dosiomics_parotid.csv",
                        "outputs/features/dosiomics_submand.csv",
                    ]
            else:
                feature_paths = [str(p) for p in args.feature_paths]

        df_feat = _load_and_merge_feature_tables(paths=feature_paths, id_col=str(args.id_col))
        df_labels = pd.read_csv(args.labels_path)[[args.id_col, args.label_col]].dropna()
        df_labels[args.id_col] = df_labels[args.id_col].astype(str)
        df_labels[args.label_col] = pd.to_numeric(df_labels[args.label_col], errors="coerce")
        df_merged = df_feat.merge(df_labels, on=str(args.id_col), how="inner")

        train_set = set(case_ids[idx_train].tolist())
        val_set = set(case_ids[idx_val].tolist())
        test_set = set(case_ids[idx_test].tolist())
        split_col = "__split__"
        df_merged[split_col] = df_merged[args.id_col].map(
            lambda x: "train" if x in train_set else ("val" if x in val_set else ("test" if x in test_set else "other"))
        )
        df_merged = df_merged[df_merged[split_col] != "other"].reset_index(drop=True)
        feature_cols = [c for c in df_feat.columns if c != str(args.id_col)]

        diag = {
            "seed": int(args.seed),
            "labels_path": str(args.labels_path),
            "feature_paths": feature_paths,
            "splits": {
                "train_path": train_path,
                "val_path": val_path,
                "test_path": test_path,
            },
            "split_stats": {
                "train": _split_stats(df_merged.loc[df_merged[split_col] == "train", args.label_col]),
                "val": _split_stats(df_merged.loc[df_merged[split_col] == "val", args.label_col]),
                "test": _split_stats(df_merged.loc[df_merged[split_col] == "test", args.label_col]),
            },
            "drift": _compute_drift(
                df=df_merged,
                id_col=str(args.id_col),
                split_col=split_col,
                feature_cols=feature_cols,
                top_k=int(args.top_k),
                ks_min_n=int(args.ks_min_n),
            ),
        }
        out_dir = os.path.dirname(diagnostics_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(diagnostics_path) and not bool(args.overwrite):
            raise FileExistsError(f"输出文件已存在：{diagnostics_path}（如需覆盖请加 --overwrite）")
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "n_total": n,
                "n_train": int(idx_train.size),
                "n_val": int(idx_val.size),
                "n_test": int(idx_test.size),
                "ratio_train": float(idx_train.size / n) if n else 0.0,
                "ratio_val": float(idx_val.size / n) if n else 0.0,
                "ratio_test": float(idx_test.size / n) if n else 0.0,
                "train_path": train_path,
                "val_path": val_path,
                "test_path": test_path,
                "diagnostics_path": diagnostics_path if bool(args.diagnostics) else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

import os
import re
import json
import time
import math
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label as cc_label

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR

CT_ROOT = os.path.join(BASE_DIR, "383nii")
DOSE_ROOT = os.path.join(BASE_DIR, "383nii")
MASK_ROOT = os.path.join(BASE_DIR, "nii")
LABELS_XLSX = os.path.join(BASE_DIR, "UniqueSamples.xlsx")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")
PLAN_JSON_ROOT = os.path.join(BASE_DIR, "383nii")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def get_case_id_from_folder(name):
    m = re.search(r"H\d{6}", name)
    if m:
        return m.group(0)
    if "-" in name:
        return name.split("-")[0]
    return name

def scan_cases(root, inner_filename, fallback_filenames=None):
    """
    扫描病例文件
    支持两种目录格式:
    1. CT383_crop格式: H229114-Cui^Bo/CT_crop.nii.gz
    2. 383nii格式: H229114_convert/H229114_CT.nii.gz
    """
    items = []
    for d in os.listdir(root):
        dp = os.path.join(root, d)
        if not os.path.isdir(dp):
            continue

        cid = get_case_id_from_folder(d)

        # 尝试多种文件名格式
        fp = None
        possible_files = []

        # 格式1: 383nii格式 - H229114_convert/H229114_CT.nii.gz
        if d.endswith('_convert'):
            base_name = d.replace('_convert', '')
            possible_files.append(os.path.join(dp, f"{base_name}_{inner_filename}"))
            # 特殊处理：CT.nii.gz 和 Dose.nii.gz
            if inner_filename == "CT.nii.gz":
                possible_files.append(os.path.join(dp, f"{base_name}_CT.nii.gz"))
            elif inner_filename == "Dose.nii.gz":
                possible_files.append(os.path.join(dp, f"{base_name}_Dose.nii.gz"))

        # 格式2: 裁剪格式 - H229114-Cui^Bo/CT_crop.nii.gz
        possible_files.append(os.path.join(dp, inner_filename))

        # 格式3: 通用格式
        possible_files.append(os.path.join(dp, f"{cid}_{inner_filename}"))

        # 添加fallback文件名
        if fallback_filenames:
            for alt in fallback_filenames:
                possible_files.append(os.path.join(dp, alt))

        # 找到第一个存在的文件
        for possible_file in possible_files:
            if os.path.exists(possible_file):
                fp = possible_file
                break

        if fp is None:
            fp = possible_files[0] if possible_files else os.path.join(dp, inner_filename)

        items.append({"case_id": cid, "folder": d, "path": fp})
    return pd.DataFrame(items)

def scan_split_masks(root, organ, side):
    items = []
    for d in os.listdir(root):
        dp = os.path.join(root, d)
        if not os.path.isdir(dp):
            continue
        fn = f"{d}_{organ}_{side}.nii.gz"
        fp = os.path.join(dp, fn)
        cid = get_case_id_from_folder(d)
        items.append({"case_id": cid, "folder": d, "path": fp})
    return pd.DataFrame(items)

def detect_case_col(df):
    cols = list(df.columns)
    score = {}
    for c in cols:
        try:
            s = df[c].astype(str)
        except:
            continue
        m = s.str.contains(r"^H\d{6}$", regex=True, na=False)
        score[c] = int(m.sum())
    if not score:
        return None
    best = max(score.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else None

def to_binary_series(s):
    v = s.copy()
    def conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            if x in [0,1]:
                return int(x)
        xs = str(x).strip().lower()
        if xs in ["1","yes","y","true","是","阳性","positive","pos"]:
            return 1
        if xs in ["0","no","n","false","否","阴性","negative","neg"]:
            return 0
        try:
            f = float(xs)
            if f in [0.0,1.0]:
                return int(f)
        except:
            pass
        return np.nan
    return v.map(conv)

def detect_label_col(df):
    for c in df.columns:
        b = to_binary_series(df[c])
        u = set(b.dropna().unique().tolist())
        if u.issubset({0,1}) and len(u) > 0:
            return c
    return None

def clean_labels(xlsx_path, out_dir):
    df0 = pd.read_excel(xlsx_path)
    c_case = detect_case_col(df0)
    c_label = detect_label_col(df0)
    if c_case is None:
        candidate = None
        for c in df0.columns:
            s = df0[c]
            if np.issubdtype(s.dtype, np.number):
                vals = s.dropna().astype(int).astype(str)
                ok = vals.str.match(r"^\d{6}$", na=False).sum()
                if ok > 0:
                    candidate = c
                    break
            else:
                vals = s.dropna().astype(str).str.strip()
                ok = vals.str.match(r"^\d{6}$", na=False).sum()
                if ok > 0:
                    candidate = c
                    break
        if candidate is None:
            raise RuntimeError("无法识别病例ID列")
        cid_series = df0[candidate].dropna().astype(int).astype(str).str.zfill(6)
        case_ids = "H" + cid_series
    else:
        case_ids = df0[c_case].astype(str)
    if c_label is None:
        raise RuntimeError("无法识别标签列")
    df = pd.DataFrame({"case_id": case_ids, "y": to_binary_series(df0[c_label])})
    df = df[df["case_id"].str.match(r"^H\d{6}$", na=False)]
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)
    df = df.drop_duplicates(subset=["case_id"], keep="first")
    ensure_dir(out_dir)
    df.to_csv(os.path.join(out_dir, "labels_clean.csv"), index=False)
    stats = {
        "total": int(df.shape[0]),
        "pos": int((df["y"]==1).sum()),
        "neg": int((df["y"]==0).sum())
    }
    with open(os.path.join(out_dir, "labels_stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return df

def build_index(ct_df, dose_df, parotid_l_df, parotid_r_df, submand_l_df, submand_r_df, out_dir):
    idx = ct_df.rename(columns={"path":"ct_path"})
    idx = idx.merge(dose_df[["case_id","path"]].rename(columns={"path":"dose_path"}), on="case_id", how="outer")
    idx = idx.merge(parotid_l_df[["case_id","path"]].rename(columns={"path":"parotid_L_mask_path"}), on="case_id", how="outer")
    idx = idx.merge(parotid_r_df[["case_id","path"]].rename(columns={"path":"parotid_R_mask_path"}), on="case_id", how="outer")
    idx = idx.merge(submand_l_df[["case_id","path"]].rename(columns={"path":"submand_L_mask_path"}), on="case_id", how="outer")
    idx = idx.merge(submand_r_df[["case_id","path"]].rename(columns={"path":"submand_R_mask_path"}), on="case_id", how="outer")
    idx["missing_ct"] = ~idx["ct_path"].apply(lambda p: isinstance(p,str) and os.path.exists(p))
    idx["missing_dose"] = ~idx["dose_path"].apply(lambda p: isinstance(p,str) and os.path.exists(p))
    idx["missing_parotid_L_mask"] = ~idx["parotid_L_mask_path"].apply(lambda p: isinstance(p,str) and os.path.exists(p))
    idx["missing_parotid_R_mask"] = ~idx["parotid_R_mask_path"].apply(lambda p: isinstance(p,str) and os.path.exists(p))
    idx["missing_submand_L_mask"] = ~idx["submand_L_mask_path"].apply(lambda p: isinstance(p,str) and os.path.exists(p))
    idx["missing_submand_R_mask"] = ~idx["submand_R_mask_path"].apply(lambda p: isinstance(p,str) and os.path.exists(p))
    ensure_dir(out_dir)
    idx.to_csv(os.path.join(out_dir, "case_file_index.csv"), index=False)
    return idx

def arrays_close(a, b):
    return np.allclose(np.array(a), np.array(b), rtol=1e-6, atol=1e-6)

def resample_to_reference(img, ref, is_mask=False):
    f = sitk.ResampleImageFilter()
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    f.SetOutputSpacing(ref.GetSpacing())
    f.SetSize(ref.GetSize())
    f.SetOutputDirection(ref.GetDirection())
    f.SetOutputOrigin(ref.GetOrigin())
    f.SetDefaultPixelValue(0)
    return f.Execute(img)

def resample_to_isotropic(img, spacing=(1.0,1.0,1.0), is_mask=False):
    s = img.GetSpacing()
    sz = img.GetSize()
    ns = tuple(float(x) for x in spacing)
    nsz = [int(math.ceil(sz[i]*s[i]/ns[i])) for i in range(3)]
    f = sitk.ResampleImageFilter()
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    f.SetOutputSpacing(ns)
    f.SetSize(nsz)
    f.SetOutputDirection(img.GetDirection())
    f.SetOutputOrigin(img.GetOrigin())
    f.SetDefaultPixelValue(0)
    return f.Execute(img)


def compute_bbox_from_masks(masks):
    """
    从多个mask计算合并的bbox
    masks: list of SimpleITK Image objects
    Returns: (xmin, ymin, zmin, xmax, ymax, zmax) or None
    """
    all_boxes = []
    ref_image = None

    for mask in masks:
        if ref_image is None:
            ref_image = mask

        arr = sitk.GetArrayFromImage(mask)
        indices = np.argwhere(arr > 0)
        if len(indices) == 0:
            continue

        # numpy使用(z,y,x)顺序，SimpleITK使用(x,y,z)
        zmin, ymin, xmin = indices.min(axis=0)
        zmax, ymax, xmax = indices.max(axis=0)
        all_boxes.append((int(xmin), int(ymin), int(zmin), int(xmax), int(ymax), int(zmax)))

    if not all_boxes:
        return None

    # 合并所有bbox
    xs = [b[0] for b in all_boxes]
    ys = [b[1] for b in all_boxes]
    zs = [b[2] for b in all_boxes]
    x2s = [b[3] for b in all_boxes]
    y2s = [b[4] for b in all_boxes]
    z2s = [b[5] for b in all_boxes]

    return (min(xs), min(ys), min(zs), max(x2s), max(y2s), max(z2s))


def crop_to_bbox(image, bbox):
    """
    裁剪图像到指定bbox
    bbox: (xmin, ymin, zmin, xmax, ymax, zmax)
    """
    x0, y0, z0, x1, y1, z1 = bbox
    size = [int(x1 - x0 + 1), int(y1 - y0 + 1), int(z1 - z0 + 1)]
    index = [int(x0), int(y0), int(z0)]
    return sitk.RegionOfInterest(image, size=size, index=index)


def crop_images_to_all_glands_bbox(ct, dose, parotid_l_mask, parotid_r_mask, submand_l_mask, submand_r_mask):
    """
    将CT、Dose裁剪到腮腺和颌下腺的合并bbox
    返回: (ct_cropped, dose_cropped)
    """
    # 计算所有腺体的合并bbox
    bbox = compute_bbox_from_masks([parotid_l_mask, parotid_r_mask, submand_l_mask, submand_r_mask])

    if bbox is None:
        # 如果无法计算bbox，返回原始图像
        return ct, dose

    # 裁剪CT和Dose
    ct_cropped = crop_to_bbox(ct, bbox)
    dose_cropped = crop_to_bbox(dose, bbox)

    return ct_cropped, dose_cropped

def ct_intensity_process(img):
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    a = np.clip(a, -1000.0, 1000.0)
    out = sitk.GetImageFromArray(a)
    out.CopyInformation(img)
    return out

def dose_intensity_process(img):
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    mx = float(a.max())
    scale = 0.01 if mx > 200.0 else 1.0
    a = a * scale
    a[a < 0.0] = 0.0
    out = sitk.GetImageFromArray(a)
    out.CopyInformation(img)
    return out

def largest_cc(mask_img):
    a = sitk.GetArrayFromImage(mask_img)
    b = (a > 0).astype(np.uint8)
    if b.sum() == 0:
        return None
    lab, n = cc_label(b)
    if n == 1:
        out = sitk.GetImageFromArray(b.astype(np.uint8))
        out.CopyInformation(mask_img)
        return out
    counts = np.bincount(lab.flatten())
    counts[0] = 0
    li = int(np.argmax(counts))
    keep = (lab == li).astype(np.uint8)
    out = sitk.GetImageFromArray(keep)
    out.CopyInformation(mask_img)
    return out

def voxel_volume(img):
    sp = img.GetSpacing()
    return float(sp[0]*sp[1]*sp[2])

def compute_dvh(dose_img, mask_img):
    d = sitk.GetArrayFromImage(dose_img).astype(np.float32)
    m = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
    vox = d[m > 0]
    if vox.size == 0:
        return {}
    vox = vox[np.isfinite(vox)]
    if vox.size == 0:
        return {}
    dv = {}
    dv["Dmean"] = float(vox.mean())
    dv["Dmin"] = float(vox.min())
    dv["Dmax"] = float(vox.max())
    dv["D2"] = float(np.percentile(vox, 98))
    dv["D5"] = float(np.percentile(vox, 95))
    dv["D50"] = float(np.percentile(vox, 50))
    dv["D95"] = float(np.percentile(vox, 5))
    for x in [1, 10, 20, 80, 90]:
        dv[f"D{x}"] = float(np.percentile(vox, 100 - x))
    vox_pos = np.clip(vox.astype(np.float64), 0.0, None)
    for a in [5.0, 10.0]:
        dv[f"EUD_a{int(a)}"] = float(np.power(np.mean(np.power(vox_pos, a)), 1.0 / a))
    hist, _ = np.histogram(vox_pos, bins=256, range=(float(vox_pos.min()), float(vox_pos.max())))
    p = hist.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    p2 = p[p > 0]
    dv["Entropy"] = float(-np.sum(p2 * np.log(p2)))
    dv["Energy"] = float(np.sum(p * p))
    for x in [5,10,20,30]:
        dv[f"V{x}"] = float((vox >= x).mean())
    return dv

def build_ct_extractor(bin_width=25.0):
    from radiomics import featureextractor
    # 按照文献标准：固定 binWidth=25HU, 重采样到 1x1x1mm
    p = {
        "binWidth": float(bin_width),
        "normalize": False,
        "correctMask": True,
        "resampledPixelSpacing": [1.0, 1.0, 1.0],
        "interpolator": "sitkBSpline",  # 图像插值
        "resamplingInterpolator": "sitkBSpline",
        "padDistance": 10,  # 保证足够的 padding
        "distances": [1, 2, 3],
    }
    ext = featureextractor.RadiomicsFeatureExtractor(**p)
    ext.enableAllFeatures()
    ext.enableImageTypes(Original={}, LoG={"sigma":[1.0, 3.0, 5.0]}, Wavelet={}) # LoG sigma 调整为更常用的 1,3,5mm
    return ext

def build_dose_extractor(bin_width=0.25):
    from radiomics import featureextractor
    # Dose单位为Gy。binWidth=0.25Gy (即 0.25Gy 一个bin，70Gy约280个bin)
    p = {
        "binWidth": float(bin_width),
        "normalize": False,
        "correctMask": True,
        "resampledPixelSpacing": [1.0, 1.0, 1.0],
        "interpolator": "sitkLinear", # Dose通常较平滑，Linear足够且不易产生负值震荡
        "resamplingInterpolator": "sitkLinear",
        "padDistance": 10,
        "distances": [1, 2, 3],
    }
    ext = featureextractor.RadiomicsFeatureExtractor(**p)
    ext.enableAllFeatures()
    ext.enableImageTypes(Original={}, LoG={"sigma":[1.0, 2.0, 3.0]}, Wavelet={})
    return ext


def compute_ct_percentiles(ct_img, mask_img):
    a = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    m = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
    vox = a[m > 0]
    vox = vox[np.isfinite(vox)]
    if vox.size == 0:
        return {}
    pcts = [5, 10, 25, 75, 90, 95]
    out = {f"P{p}": float(np.percentile(vox, p)) for p in pcts}
    out["P95_P5"] = float(out["P95"] - out["P5"])
    return out


def compute_dose_gradient(dose_img, mask_img):
    d = sitk.GetArrayFromImage(dose_img).astype(np.float32)
    m = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
    sp = tuple(float(x) for x in dose_img.GetSpacing())
    dz, dy, dx = np.gradient(d, sp[2], sp[1], sp[0])
    g = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    vox = g[m > 0]
    vox = vox[np.isfinite(vox)]
    if vox.size == 0:
        return {}
    return {
        "mean": float(vox.mean()),
        "std": float(vox.std(ddof=1)) if vox.size > 1 else 0.0,
        "p90": float(np.percentile(vox, 90)),
        "p95": float(np.percentile(vox, 95)),
        "max": float(vox.max()),
    }

def extract_for_one_mask(ct_ref, dose_ref, ct_i, dose_i, mask, organ_prefix, target_spacing, use_largest_cc=True):
    # 1. 确保 mask 与 CT 参考空间对齐 (物理空间对齐)
    if (not arrays_close(ct_ref.GetSpacing(), mask.GetSpacing()) or
        not arrays_close(ct_ref.GetOrigin(), mask.GetOrigin()) or
        not arrays_close(ct_ref.GetDirection(), mask.GetDirection())):
        mask = resample_to_reference(mask, ct_ref, is_mask=True)
    
    # 2. 这里的重采样是为了计算 DVH 和体积 (使用 target_spacing)
    # PyRadiomics 内部会再次根据 resampledPixelSpacing 进行重采样用于特征提取
    mask_i = resample_to_isotropic(mask, spacing=target_spacing, is_mask=True)
    
    if use_largest_cc:
        mask_i = largest_cc(mask_i)
    if mask_i is None:
        return {f"{organ_prefix}__mask_nonempty": 0}
        
    vxl = voxel_volume(ct_i) # 注意：这里 voxel_volume 可能不准确，因为 ct_i 没有重采样，但 mask_i 重采样了
    # 修正：体积计算应该基于 mask_i 的 spacing
    mv = int(sitk.GetArrayFromImage(mask_i).sum())
    # 重新计算 voxel volume 基于 mask_i
    sp = mask_i.GetSpacing()
    vol_mm3 = float(mv) * float(sp[0]*sp[1]*sp[2])
    
    # 计算 DVH (需要 mask 和 dose 在同一空间)
    # dose_i 还没有重采样到 target_spacing，这可能导致 DVH 计算不匹配
    # 为了准确 DVH，应该把 dose 也重采样到 mask_i 的空间
    dose_resampled = resample_to_reference(dose_i, mask_i, is_mask=False)
    dvh = compute_dvh(dose_resampled, mask_i)
    ct_resampled_for_stats = resample_to_reference(ct_i, mask_i, is_mask=False)
    ct_pct = compute_ct_percentiles(ct_resampled_for_stats, mask_i)
    dose_grad = compute_dose_gradient(dose_resampled, mask_i)
    
    # 3. 特征提取
    # 严格使用固定 binWidth，不动态计算！
    # CT: 25 HU
    # Dose: 0.25 Gy
    
    # 注意：PyRadiomics 需要 mask 和 image 在同一几何空间（或者能自动对齐）
    # 我们传入原始的 ct_i / dose_i 和 经过对齐处理的 mask (但在 PyRadiomics 内部会再次重采样)
    # 为了保险，最好传入物理空间一致的 mask。
    # 上面的 mask 已经被 resample_to_reference 到 ct_ref (即 ct_i) 了吗？
    # L441: mask = resample_to_reference(mask, ct_ref) -> 此时 mask 和 ct_i 空间一致
    # L442: mask_i = resample_to_isotropic(...) -> 此时 mask_i 空间变了
    
    # PyRadiomics 最佳实践：传入原始 Image 和 与原始 Image 空间对齐的 Mask
    # 所以我们应该使用 L441 得到的 `mask` (与 CT 对齐)，而不是 `mask_i` (重采样后的)
    # 让 PyRadiomics 自己去做 resampledPixelSpacing=[1,1,1]
    
    # 重新确保 mask 与 ct_i 对齐
    mask_for_features = mask # 这是已经对齐到 ct_ref 的
    
    ct_ext = build_ct_extractor(bin_width=25.0)
    dose_ext = build_dose_extractor(bin_width=0.25)
    
    try:
        # 修改：传入原始 CT (ct_ref) 而不是手动重采样的 ct_i
        # PyRadiomics 会根据配置 (resampledPixelSpacing=[1,1,1], interpolator=sitkBSpline) 自动进行 BSpline 重采样
        # 避免了手动 Linear 重采样导致的纹理模糊和二次插值问题
        ct_res = ct_ext.execute(ct_ref, mask_for_features, label=1)
    except Exception as e:
        print(f"CT extraction failed: {e}")
        ct_res = {}

    try:
        # 修改：传入原始 Dose (dose_ref)
        # PyRadiomics 会根据配置 (resampledPixelSpacing=[1,1,1], interpolator=sitkLinear) 自动重采样
        dose_res = dose_ext.execute(dose_ref, mask_for_features, label=1)
    except Exception as e:
        print(f"Dose extraction failed: {e}")
        dose_res = {}
        
    out = {
        f"{organ_prefix}__mask_nonempty": 1,
        f"{organ_prefix}__roi_voxels": mv,
        f"{organ_prefix}__roi_volume_mm3": vol_mm3
    }
    for k, v in ct_res.items():
        try:
            vv = float(np.asarray(v))
            if not math.isnan(vv):
                out[f"{organ_prefix}__ct__{k}"] = vv
        except:
            pass
    for k, v in dose_res.items():
        try:
            vv = float(np.asarray(v))
            if not math.isnan(vv):
                out[f"{organ_prefix}__dose__{k}"] = vv
        except:
            pass
    for k, v in dvh.items():
        out[f"{organ_prefix}__dvh__{k}"] = float(v)
    for k, v in ct_pct.items():
        out[f"{organ_prefix}__ct_extra__{k}"] = float(v)
    for k, v in dose_grad.items():
        out[f"{organ_prefix}__dose_grad__{k}"] = float(v)
    return out

def _add_symmetric_summary_features(out, organ, left_prefix, right_prefix):
    l_pref = f"{left_prefix}__"
    r_pref = f"{right_prefix}__"
    candidates = [k for k in out.keys() if k.startswith(l_pref)]
    eps = 1e-8
    for lk in candidates:
        suffix = lk[len(l_pref):]
        rk = r_pref + suffix
        if rk not in out:
            continue
        lv = out.get(lk, None)
        rv = out.get(rk, None)
        try:
            lvf = float(lv)
            rvf = float(rv)
        except:
            continue
        if math.isnan(lvf) or math.isnan(rvf):
            continue
        out[f"{organ}__sym__mean__{suffix}"] = (lvf + rvf) / 2.0
        out[f"{organ}__sym__max__{suffix}"] = max(lvf, rvf)
        out[f"{organ}__sym__min__{suffix}"] = min(lvf, rvf)
        out[f"{organ}__sym__absdiff__{suffix}"] = abs(lvf - rvf)

        if _sym_extra_ok(suffix):
            out[f"{organ}__sym__diff__{suffix}"] = (lvf - rvf)
            denom = max(abs(lvf), abs(rvf), eps)
            out[f"{organ}__sym__reldiff__{suffix}"] = abs(lvf - rvf) / denom
            abs_l = abs(lvf)
            abs_r = abs(rvf)
            abs_max = max(abs_l, abs_r)
            abs_min = min(abs_l, abs_r)
            out[f"{organ}__sym__absratio__{suffix}"] = abs_max / max(abs_min, eps)

def _sym_extra_ok(suffix):
    if suffix.startswith("roi_"):
        return True
    if "__dvh__" in suffix:
        return True
    if suffix.startswith("ct__original_") or suffix.startswith("dose__original_"):
        return True
    return False

def _augment_symmetric_features_df(df, organ, left_prefix, right_prefix):
    l_pref = f"{left_prefix}__"
    r_pref = f"{right_prefix}__"
    eps = 1e-8
    l_cols = [c for c in df.columns if c.startswith(l_pref)]
    if not l_cols:
        return df, 0
    added = 0
    for lc in l_cols:
        suffix = lc[len(l_pref):]
        rc = r_pref + suffix
        if rc not in df.columns:
            continue
        lv = pd.to_numeric(df[lc], errors="coerce")
        rv = pd.to_numeric(df[rc], errors="coerce")
        mean_col = f"{organ}__sym__mean__{suffix}"
        max_col = f"{organ}__sym__max__{suffix}"
        min_col = f"{organ}__sym__min__{suffix}"
        absdiff_col = f"{organ}__sym__absdiff__{suffix}"
        diff_col = f"{organ}__sym__diff__{suffix}"
        reldiff_col = f"{organ}__sym__reldiff__{suffix}"
        absratio_col = f"{organ}__sym__absratio__{suffix}"

        if mean_col not in df.columns:
            df[mean_col] = (lv + rv) / 2.0
            added += 1
        if max_col not in df.columns:
            df[max_col] = pd.concat([lv, rv], axis=1).max(axis=1)
            added += 1
        if min_col not in df.columns:
            df[min_col] = pd.concat([lv, rv], axis=1).min(axis=1)
            added += 1
        if absdiff_col not in df.columns:
            df[absdiff_col] = (lv - rv).abs()
            added += 1
        if _sym_extra_ok(suffix):
            if diff_col not in df.columns:
                df[diff_col] = (lv - rv)
                added += 1
            if reldiff_col not in df.columns:
                denom = pd.concat([lv.abs(), rv.abs()], axis=1).max(axis=1).clip(lower=eps)
                df[reldiff_col] = (lv - rv).abs() / denom
                added += 1
            if absratio_col not in df.columns:
                abs_lv = lv.abs()
                abs_rv = rv.abs()
                abs_max = pd.concat([abs_lv, abs_rv], axis=1).max(axis=1)
                abs_min = pd.concat([abs_lv, abs_rv], axis=1).min(axis=1).clip(lower=eps)
                df[absratio_col] = abs_max / abs_min
                added += 1
    return df, added

def augment_symmetric_features_in_features_dir():
    feat_dir = os.path.join(OUTPUT_ROOT, "features")
    targets = [
        "ct_radiomics.csv",
        "dosiomics.csv",
        "dvh.csv",
        "ct_radiomics_parotid.csv",
        "ct_radiomics_submand.csv",
        "dosiomics_parotid.csv",
        "dosiomics_submand.csv",
        "ct_radiomics_raw.csv",
        "dosiomics_raw.csv",
        "dvh_raw.csv",
    ]
    total_added = {}
    for name in targets:
        path = os.path.join(feat_dir, name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={"case_id": str})
        before_cols = int(df.shape[1])
        df, a1 = _augment_symmetric_features_df(df, "parotid", "parotid_L", "parotid_R")
        df, a2 = _augment_symmetric_features_df(df, "submand", "submand_L", "submand_R")
        after_cols = int(df.shape[1])
        if after_cols != before_cols:
            df.to_csv(path, index=False)
        total_added[name] = int(a1 + a2)
    return total_added

def extract_features_for_case(rec, target_spacing=(1.0,1.0,1.0), save_cropped=False, ct_crop_dir=None, dose_crop_dir=None):
    if (
        rec.get("missing_ct", False)
        or rec.get("missing_dose", False)
        or rec.get("missing_parotid_L_mask", False)
        or rec.get("missing_parotid_R_mask", False)
        or rec.get("missing_submand_L_mask", False)
        or rec.get("missing_submand_R_mask", False)
    ):
        return None
    ct = sitk.ReadImage(rec["ct_path"])
    dose = sitk.ReadImage(rec["dose_path"])
    parotid_l_mask = sitk.ReadImage(rec["parotid_L_mask_path"])
    parotid_r_mask = sitk.ReadImage(rec["parotid_R_mask_path"])
    submand_l_mask = sitk.ReadImage(rec["submand_L_mask_path"])
    submand_r_mask = sitk.ReadImage(rec["submand_R_mask_path"])

    # 新增：裁剪到腮腺和颌下腺bbox（如果使用原始383nii数据）
    # 检查是否使用convert目录（原始数据），如果是则进行bbox裁剪
    cropped = False
    if '_convert' in rec.get("ct_path", ""):
        ct, dose = crop_images_to_all_glands_bbox(ct, dose, parotid_l_mask, parotid_r_mask, submand_l_mask, submand_r_mask)
        cropped = True

        # 保存裁剪后的文件
        if save_cropped and ct_crop_dir and dose_crop_dir:
            case_id = rec["case_id"]
            ensure_dir(ct_crop_dir)
            ensure_dir(dose_crop_dir)

            ct_crop_path = os.path.join(ct_crop_dir, f"{case_id}_CT_crop.nii.gz")
            dose_crop_path = os.path.join(dose_crop_dir, f"{case_id}_Dose_crop.nii.gz")

            sitk.WriteImage(ct, ct_crop_path)
            sitk.WriteImage(dose, dose_crop_path)

    if (not arrays_close(ct.GetSpacing(), dose.GetSpacing()) or
        not arrays_close(ct.GetOrigin(), dose.GetOrigin()) or
        not arrays_close(ct.GetDirection(), dose.GetDirection())):
        dose = resample_to_reference(dose, ct, is_mask=False)
    ct_i = resample_to_isotropic(ct, spacing=target_spacing, is_mask=False)
    dose_i = resample_to_isotropic(dose, spacing=target_spacing, is_mask=False)
    ct_i = ct_intensity_process(ct_i)
    dose_i = dose_intensity_process(dose_i)
    out = {"case_id": rec["case_id"]}
    out.update(extract_for_one_mask(ct, dose, ct_i, dose_i, parotid_l_mask, "parotid_L", target_spacing))
    out.update(extract_for_one_mask(ct, dose, ct_i, dose_i, parotid_r_mask, "parotid_R", target_spacing))
    out.update(extract_for_one_mask(ct, dose, ct_i, dose_i, submand_l_mask, "submand_L", target_spacing))
    out.update(extract_for_one_mask(ct, dose, ct_i, dose_i, submand_r_mask, "submand_R", target_spacing))
    _add_symmetric_summary_features(out, "parotid", "parotid_L", "parotid_R")
    _add_symmetric_summary_features(out, "submand", "submand_L", "submand_R")
    return out

def flatten_feature_dicts(rows, key):
    items = []
    for r in rows:
        if r is None:
            continue
        d = r.get(key, {})
        d2 = {"case_id": r["case_id"]}
        for k, v in d.items():
            d2[k] = v
        items.append(d2)
    if not items:
        return pd.DataFrame(columns=["case_id"])
    return pd.DataFrame(items)

def run(limit=None, save_cropped=False, ct_crop_dir=None, dose_crop_dir=None):
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("radiomics").setLevel(logging.WARNING)
    logging.getLogger("radiomics.imageoperations").setLevel(logging.WARNING)
    logging.getLogger("radiomics.featureextractor").setLevel(logging.WARNING)

    t0 = time.time()
    ensure_dir(OUTPUT_ROOT)
    cur_dir = os.path.join(OUTPUT_ROOT, "curation")
    feat_dir = os.path.join(OUTPUT_ROOT, "features")
    ensure_dir(cur_dir)
    ensure_dir(feat_dir)

    # 创建裁剪输出目录
    if save_cropped:
        if ct_crop_dir is None:
            ct_crop_dir = os.path.join(BASE_DIR, "CT_crop")
        if dose_crop_dir is None:
            dose_crop_dir = os.path.join(BASE_DIR, "Dose_crop")
        ensure_dir(ct_crop_dir)
        ensure_dir(dose_crop_dir)
        print(f"裁剪文件将保存到:")
        print(f"  CT: {ct_crop_dir}")
        print(f"  Dose: {dose_crop_dir}")

    labels_df = clean_labels(LABELS_XLSX, cur_dir)
    ct_df = scan_cases(CT_ROOT, "CT.nii.gz", fallback_filenames=["CT_crop.nii.gz"])
    dose_df = scan_cases(DOSE_ROOT, "Dose.nii.gz", fallback_filenames=["Dose_crop.nii.gz", "RD_dose.nii.gz"])
    parotid_l_df = scan_split_masks(MASK_ROOT, "parotid", "L")
    parotid_r_df = scan_split_masks(MASK_ROOT, "parotid", "R")
    submand_l_df = scan_split_masks(MASK_ROOT, "submand", "L")
    submand_r_df = scan_split_masks(MASK_ROOT, "submand", "R")
    idx = build_index(ct_df, dose_df, parotid_l_df, parotid_r_df, submand_l_df, submand_r_df, cur_dir)
    idx2 = idx.merge(labels_df, on="case_id", how="left")
    idx2.to_csv(os.path.join(cur_dir, "index_with_labels.csv"), index=False)
    valid = idx2[
        (~idx2["missing_ct"])
        & (~idx2["missing_dose"])
        & (~idx2["missing_parotid_L_mask"])
        & (~idx2["missing_parotid_R_mask"])
        & (~idx2["missing_submand_L_mask"])
        & (~idx2["missing_submand_R_mask"])
        & (~idx2["y"].isna())
    ]
    if limit is not None:
        valid = valid.head(int(limit))

    # Optimization: Use joblib for parallel extraction
    from joblib import Parallel, delayed
    import multiprocessing
    from tqdm import tqdm
    n_jobs = multiprocessing.cpu_count()
    print(f"正在使用 {n_jobs} 个核心并行处理...")

    success_count = 0
    failure_count = 0

    def _process_one_case(rec):
        r = {
            "case_id": rec["case_id"],
            "ct_path": rec["ct_path"],
            "dose_path": rec["dose_path"],
            "parotid_L_mask_path": rec["parotid_L_mask_path"],
            "parotid_R_mask_path": rec["parotid_R_mask_path"],
            "submand_L_mask_path": rec["submand_L_mask_path"],
            "submand_R_mask_path": rec["submand_R_mask_path"],
            "missing_ct": bool(rec["missing_ct"]),
            "missing_dose": bool(rec["missing_dose"]),
            "missing_parotid_L_mask": bool(rec["missing_parotid_L_mask"]),
            "missing_parotid_R_mask": bool(rec["missing_parotid_R_mask"]),
            "missing_submand_L_mask": bool(rec["missing_submand_L_mask"]),
            "missing_submand_R_mask": bool(rec["missing_submand_R_mask"]),
        }
        try:
            result = extract_features_for_case(r, target_spacing=(1.0,1.0,1.0),
                                                save_cropped=save_cropped,
                                                ct_crop_dir=ct_crop_dir,
                                                dose_crop_dir=dose_crop_dir)
            return r["case_id"], result, None
        except Exception as e:
            return r["case_id"], None, str(e)

    rows = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_process_one_case)(rec) for _, rec in tqdm(valid.iterrows(), total=len(valid), desc="特征提取并行处理")
    )

    for case_id, result, error in rows:
        if result is not None:
            success_count += 1
        else:
            failure_count += 1
            if error:
                tqdm.write(f"错误: {case_id} - {error}")

    print(f"\n并行处理完成: 成功 {success_count}, 失败 {failure_count}")

    qc = []
    flat_rows = []
    for case_id, result, error in rows:
        if result is not None:
            flat_rows.append(result)
            m_pl = int(result.get("parotid_L__mask_nonempty", 0))
            m_pr = int(result.get("parotid_R__mask_nonempty", 0))
            m_sl = int(result.get("submand_L__mask_nonempty", 0))
            m_sr = int(result.get("submand_R__mask_nonempty", 0))
            m_p = int((m_pl == 1) and (m_pr == 1))
            m_s = int((m_sl == 1) and (m_sr == 1))
            qc.append({
                "case_id": result["case_id"],
                "mask_nonempty": int((m_p == 1) and (m_s == 1)),
                "parotid__mask_nonempty": m_p,
                "submand__mask_nonempty": m_s,
                "parotid_L__mask_nonempty": m_pl,
                "parotid_R__mask_nonempty": m_pr,
                "submand_L__mask_nonempty": m_sl,
                "submand_R__mask_nonempty": m_sr,
            })
    qc_df = pd.DataFrame(qc)
    qc_df.to_csv(os.path.join(feat_dir, "roi_qc.csv"), index=False)
    if flat_rows:
        full_df = pd.DataFrame(flat_rows)
    else:
        full_df = pd.DataFrame(columns=["case_id"])

    print("\n正在保存特征文件...")
    ct_cols = ["case_id"] + [c for c in full_df.columns if "__ct__" in c]
    dose_cols = ["case_id"] + [c for c in full_df.columns if "__dose__" in c]
    dvh_cols = ["case_id"] + [c for c in full_df.columns if "__dvh__" in c]
    ct_tab = full_df[ct_cols]
    dose_tab = full_df[dose_cols]
    dvh_tab = full_df[dvh_cols]
    print("  - 保存原始特征文件...")
    ct_tab.to_csv(os.path.join(feat_dir, "ct_radiomics_raw.csv"), index=False)
    dose_tab.to_csv(os.path.join(feat_dir, "dosiomics_raw.csv"), index=False)
    dvh_tab.to_csv(os.path.join(feat_dir, "dvh_raw.csv"), index=False)
    ct_parotid_cols = ["case_id"] + [c for c in full_df.columns if c.startswith("parotid") and "__ct__" in c]
    ct_submand_cols = ["case_id"] + [c for c in full_df.columns if c.startswith("submand") and "__ct__" in c]
    dose_parotid_cols = ["case_id"] + [c for c in full_df.columns if c.startswith("parotid") and "__dose__" in c]
    dose_submand_cols = ["case_id"] + [c for c in full_df.columns if c.startswith("submand") and "__dose__" in c]
    print("  - 生成分拆特征...")
    ct_parotid_tab = full_df[ct_parotid_cols]
    ct_submand_tab = full_df[ct_submand_cols]
    dose_parotid_tab = full_df[dose_parotid_cols]
    dose_submand_tab = full_df[dose_submand_cols]
    ct_final = ct_tab.merge(labels_df, on="case_id", how="left").merge(qc_df[["case_id","mask_nonempty"]], on="case_id", how="left")
    dose_final = dose_tab.merge(labels_df, on="case_id", how="left").merge(qc_df[["case_id","mask_nonempty"]], on="case_id", how="left")
    dvh_final = dvh_tab.merge(labels_df, on="case_id", how="left").merge(qc_df[["case_id","mask_nonempty"]], on="case_id", how="left")
    ct_final = ct_final[(ct_final["mask_nonempty"]==1) & (~ct_final["y"].isna())]
    dose_final = dose_final[(dose_final["mask_nonempty"]==1) & (~dose_final["y"].isna())]
    dvh_final = dvh_final[(dvh_final["mask_nonempty"]==1) & (~dvh_final["y"].isna())]
    ct_parotid_final = ct_parotid_tab.merge(labels_df, on="case_id", how="left").merge(qc_df[["case_id","parotid__mask_nonempty"]], on="case_id", how="left")
    ct_submand_final = ct_submand_tab.merge(labels_df, on="case_id", how="left").merge(qc_df[["case_id","submand__mask_nonempty"]], on="case_id", how="left")
    dose_parotid_final = dose_parotid_tab.merge(labels_df, on="case_id", how="left").merge(qc_df[["case_id","parotid__mask_nonempty"]], on="case_id", how="left")
    dose_submand_final = dose_submand_tab.merge(labels_df, on="case_id", how="left").merge(qc_df[["case_id","submand__mask_nonempty"]], on="case_id", how="left")
    ct_parotid_final = ct_parotid_final[(ct_parotid_final["parotid__mask_nonempty"]==1) & (~ct_parotid_final["y"].isna())]
    ct_submand_final = ct_submand_final[(ct_submand_final["submand__mask_nonempty"]==1) & (~ct_submand_final["y"].isna())]
    dose_parotid_final = dose_parotid_final[(dose_parotid_final["parotid__mask_nonempty"]==1) & (~dose_parotid_final["y"].isna())]
    dose_submand_final = dose_submand_final[(dose_submand_final["submand__mask_nonempty"]==1) & (~dose_submand_final["y"].isna())]
    ct_parotid_final = ct_parotid_final[[c for c in ct_parotid_final.columns if ("diagnostics_" not in c and c != "parotid__mask_nonempty")]]
    ct_submand_final = ct_submand_final[[c for c in ct_submand_final.columns if ("diagnostics_" not in c and c != "submand__mask_nonempty")]]
    dose_parotid_final = dose_parotid_final[[c for c in dose_parotid_final.columns if ("diagnostics_" not in c and c != "parotid__mask_nonempty")]]
    dose_submand_final = dose_submand_final[[c for c in dose_submand_final.columns if ("diagnostics_" not in c and c != "submand__mask_nonempty")]]
    print("  - 保存最终特征文件...")
    ct_final.to_csv(os.path.join(feat_dir, "ct_radiomics.csv"), index=False)
    dose_final.to_csv(os.path.join(feat_dir, "dosiomics.csv"), index=False)
    dvh_final.to_csv(os.path.join(feat_dir, "dvh.csv"), index=False)
    ct_parotid_final.to_csv(os.path.join(feat_dir, "ct_radiomics_parotid.csv"), index=False)
    ct_submand_final.to_csv(os.path.join(feat_dir, "ct_radiomics_submand.csv"), index=False)
    dose_parotid_final.to_csv(os.path.join(feat_dir, "dosiomics_parotid.csv"), index=False)
    dose_submand_final.to_csv(os.path.join(feat_dir, "dosiomics_submand.csv"), index=False)
    t1 = time.time()
    meta = {
        "runtime_sec": float(t1-t0),
        "cases_processed": int(len(rows)),
        "ct_features_cols": int(ct_tab.shape[1]),
        "dose_features_cols": int(dose_tab.shape[1]),
        "dvh_cols": int(dvh_tab.shape[1])
    }
    with open(os.path.join(feat_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def prepare_ml_tables():
    feat_dir = os.path.join(OUTPUT_ROOT, "features")
    ensure_dir(feat_dir)
    ct_path = os.path.join(feat_dir, "ct_radiomics.csv")
    dose_path = os.path.join(feat_dir, "dosiomics.csv")
    dvh_path = os.path.join(feat_dir, "dvh.csv")
    def _drop_constant_cols(df, key_cols):
        cols = [c for c in df.columns if c not in key_cols]
        drop = []
        for c in cols:
            s = df[c]
            try:
                v = s.dropna()
                if v.size == 0:
                    drop.append(c)
                    continue
                if v.nunique() <= 1:
                    drop.append(c)
                    continue
                vc = v.value_counts(normalize=True)
                ratio = float(vc.iloc[0]) if vc.size > 0 else 1.0
                if ratio >= 0.99:
                    drop.append(c)
            except:
                pass
        if drop:
            df = df.drop(columns=drop)
        return df, drop
    if os.path.exists(ct_path):
        ct_df = pd.read_csv(ct_path)
        keep = []
        for c in ct_df.columns:
            # diagnostics usually not needed for ML, but user requested NO filtering. 
            # We keep them or drop them? "只提取，不筛选" usually refers to feature selection.
            # Diagnostics are metadata. I will keep dropping diagnostics as they are not "features".
            if c.startswith("diagnostics_"):
                continue
            keep.append(c)
        ct_df = ct_df[keep]
        if "y" in ct_df.columns:
            ct_df["y"] = ct_df["y"].astype(int)
        if "mask_nonempty" in ct_df.columns:
            ct_df = ct_df.drop(columns=["mask_nonempty"])
        # REMOVED: _drop_constant_cols
        ct_df.to_csv(os.path.join(feat_dir, "ct_radiomics_ml.csv"), index=False)
        ct_ml_cols = int(ct_df.shape[1])
        ct_const_drop = 0
    else:
        ct_ml_cols = 0
        ct_const_drop = 0
    if os.path.exists(dose_path):
        dose_df = pd.read_csv(dose_path)
        keep = []
        for c in dose_df.columns:
            if c.startswith("diagnostics_"):
                continue
            if "shape_" in c:
                continue
            keep.append(c)
        dose_df = dose_df[keep]
        if "y" in dose_df.columns:
            dose_df["y"] = dose_df["y"].astype(int)
        if "mask_nonempty" in dose_df.columns:
            dose_df = dose_df.drop(columns=["mask_nonempty"])
        # REMOVED: _drop_constant_cols
        dose_df.to_csv(os.path.join(feat_dir, "dosiomics_ml.csv"), index=False)
        dose_ml_cols = int(dose_df.shape[1])
        dose_const_drop = 0
    else:
        dose_ml_cols = 0
        dose_const_drop = 0
    if os.path.exists(dvh_path):
        dvh_df = pd.read_csv(dvh_path)
        if "y" in dvh_df.columns:
            dvh_df["y"] = dvh_df["y"].astype(int)
        if "mask_nonempty" in dvh_df.columns:
            dvh_df = dvh_df.drop(columns=["mask_nonempty"])
        # REMOVED: _drop_constant_cols
        dvh_df.to_csv(os.path.join(feat_dir, "dvh_ml.csv"), index=False)
        dvh_ml_cols = int(dvh_df.shape[1])
        dvh_const_drop = 0
    else:
        dvh_ml_cols = 0
        dvh_const_drop = 0
    ml_meta = {
        "ct_ml_cols": ct_ml_cols,
        "dose_ml_cols": dose_ml_cols,
        "dvh_ml_cols": dvh_ml_cols,
        "ct_constants_dropped": ct_const_drop,
        "dose_constants_dropped": dose_const_drop,
        "dvh_constants_dropped": dvh_const_drop
    }
    with open(os.path.join(feat_dir, "ml_meta.json"), "w") as f:
        json.dump(ml_meta, f, ensure_ascii=False, indent=2)

def _safe_float_series(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except:
        return pd.Series([np.nan]*len(s))

def run_qc_checks(n_sample=5):
    feat_dir = os.path.join(OUTPUT_ROOT, "features")
    cur_dir = os.path.join(OUTPUT_ROOT, "curation")
    ensure_dir(feat_dir)
    ct_raw_path = os.path.join(feat_dir, "ct_radiomics_raw.csv")
    dose_raw_path = os.path.join(feat_dir, "dosiomics_raw.csv")
    dvh_raw_path = os.path.join(feat_dir, "dvh_raw.csv")
    idx_path = os.path.join(cur_dir, "case_file_index.csv")
    rep = {}
    if os.path.exists(ct_raw_path):
        ct_df = pd.read_csv(ct_raw_path)
    else:
        ct_df = None
    if os.path.exists(dose_raw_path):
        dose_df = pd.read_csv(dose_raw_path)
    else:
        dose_df = None
    if os.path.exists(dvh_raw_path):
        dvh_df = pd.read_csv(dvh_raw_path)
    else:
        dvh_df = None
    if ct_df is not None:
        ct_min_cols = [c for c in ct_df.columns if "__ct__original_firstorder_Minimum" in c]
        ct_max_cols = [c for c in ct_df.columns if "__ct__original_firstorder_Maximum" in c]
        ct_min_vals = []
        ct_max_vals = []
        for c in ct_min_cols:
            ct_min_vals.extend(_safe_float_series(ct_df[c]).dropna().tolist())
        for c in ct_max_cols:
            ct_max_vals.extend(_safe_float_series(ct_df[c]).dropna().tolist())
        if ct_min_vals:
            rep["ct_hu_min_min"] = float(np.min(ct_min_vals))
            rep["ct_hu_min_p1"] = float(np.percentile(ct_min_vals, 1))
        else:
            rep["ct_hu_min_min"] = None
            rep["ct_hu_min_p1"] = None
        if ct_max_vals:
            rep["ct_hu_max_max"] = float(np.max(ct_max_vals))
            rep["ct_hu_max_p99"] = float(np.percentile(ct_max_vals, 99))
        else:
            rep["ct_hu_max_max"] = None
            rep["ct_hu_max_p99"] = None
        rep["ct_hu_cols_checked"] = {"min_cols": ct_min_cols, "max_cols": ct_max_cols}
    if dose_df is not None:
        dose_max_cols = [c for c in dose_df.columns if "__dose__original_firstorder_Maximum" in c]
        dose_max_vals = []
        for c in dose_max_cols:
            dose_max_vals.extend(_safe_float_series(dose_df[c]).dropna().tolist())
        if dose_max_vals:
            rep["dose_max_min"] = float(np.min(dose_max_vals))
            rep["dose_max_mean"] = float(np.mean(dose_max_vals))
            rep["dose_max_median"] = float(np.median(dose_max_vals))
            rep["dose_max_max"] = float(np.max(dose_max_vals))
            rep["dose_max_p95"] = float(np.percentile(dose_max_vals, 95))
            rep["dose_max_p99"] = float(np.percentile(dose_max_vals, 99))
            rep["dose_max_cols_checked"] = dose_max_cols
            rep["dose_max_mean_minus_70"] = float(rep["dose_max_mean"] - 70.0)
        else:
            rep["dose_max_min"] = None
            rep["dose_max_mean"] = None
            rep["dose_max_median"] = None
            rep["dose_max_max"] = None
            rep["dose_max_p95"] = None
            rep["dose_max_p99"] = None
            rep["dose_max_cols_checked"] = []
            rep["dose_max_mean_minus_70"] = None
    inf_nan = {}
    def _inf_nan_counts(df, prefix):
        cols = [c for c in df.columns if c != "case_id"]
        vals = df[cols].apply(pd.to_numeric, errors="coerce")
        n_nan = int(vals.isna().sum().sum())
        v = vals.values
        n_inf = int(np.isinf(v).sum())
        lc = [c.lower() for c in cols]
        sel_wave = [c for c in cols if "wavelet" in c.lower()]
        sel_log = [c for c in cols if ("log" in c.lower() or "log-sigma" in c.lower() or "log_" in c.lower() or "logsigma" in c.lower() or "logsigma" in c)]
        vals_wave = df[sel_wave].apply(pd.to_numeric, errors="coerce") if sel_wave else pd.DataFrame()
        vals_log = df[sel_log].apply(pd.to_numeric, errors="coerce") if sel_log else pd.DataFrame()
        n_nan_wave = int(vals_wave.isna().sum().sum()) if not vals_wave.empty else 0
        n_inf_wave = int(np.isinf(vals_wave.values).sum()) if not vals_wave.empty else 0
        n_nan_log = int(vals_log.isna().sum().sum()) if not vals_log.empty else 0
        n_inf_log = int(np.isinf(vals_log.values).sum()) if not vals_log.empty else 0
        return {
            "total_nan": n_nan,
            "total_inf": n_inf,
            "wavelet_nan": n_nan_wave,
            "wavelet_inf": n_inf_wave,
            "log_nan": n_nan_log,
            "log_inf": n_inf_log,
            "wavelet_cols": sel_wave,
            "log_cols": sel_log,
        }
    if ct_df is not None:
        inf_nan["ct"] = _inf_nan_counts(ct_df, "ct")
    if dose_df is not None:
        inf_nan["dose"] = _inf_nan_counts(dose_df, "dose")
    rep["inf_nan"] = inf_nan
    std_comp = {}
    def _std_cols(df, key):
        cols = []
        for c in df.columns:
            lc = c.lower()
            if "firstorder_standarddeviation" in lc:
                cols.append(c)
        return cols
    if ct_df is not None:
        ct_std_cols = _std_cols(ct_df, "ct")
        ct_std_vals = []
        for c in ct_std_cols:
            ct_std_vals.extend(_safe_float_series(ct_df[c]).dropna().tolist())
        if ct_std_vals:
            std_comp["ct_std_mean"] = float(np.mean(ct_std_vals))
            std_comp["ct_std_median"] = float(np.median(ct_std_vals))
            std_comp["ct_std_p95"] = float(np.percentile(ct_std_vals, 95))
            std_comp["ct_std_cols"] = ct_std_cols
    if dose_df is not None:
        dose_std_cols = _std_cols(dose_df, "dose")
        dose_std_vals = []
        for c in dose_std_cols:
            dose_std_vals.extend(_safe_float_series(dose_df[c]).dropna().tolist())
        if dose_std_vals:
            std_comp["dose_std_mean"] = float(np.mean(dose_std_vals))
            std_comp["dose_std_median"] = float(np.median(dose_std_vals))
            std_comp["dose_std_p95"] = float(np.percentile(dose_std_vals, 95))
            std_comp["dose_std_cols"] = dose_std_cols
    rep["std_compare_ct_vs_dose"] = std_comp
    def _column_std_distribution(df):
        cols = [c for c in df.columns if c != "case_id"]
        vals = df[cols].apply(pd.to_numeric, errors="coerce")
        s = vals.std(axis=0, ddof=1)
        s = s.dropna()
        if s.size == 0:
            return {"mean": None, "median": None, "p95": None, "count": 0}
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p95": float(np.percentile(s.values, 95)),
            "count": int(s.size),
        }
    if ct_df is not None:
        std_comp["ct_feature_std_dist"] = _column_std_distribution(ct_df)
    if dose_df is not None:
        std_comp["dose_feature_std_dist"] = _column_std_distribution(dose_df)
    wave_log_std = {}
    def _wave_log_std_anomaly(df):
        out = {}
        cols = [c for c in df.columns if c != "case_id"]
        wave_cols = [c for c in cols if "wavelet" in c.lower() and "standarddeviation" in c.lower()]
        log_cols = [c for c in cols if ("log" in c.lower() or "log-sigma" in c.lower()) and "standarddeviation" in c.lower()]
        def _anomaly_counts(sub_df):
            vals = sub_df.apply(pd.to_numeric, errors="coerce")
            items = []
            for c in vals.columns:
                s = vals[c].dropna().values
                if s.size == 0:
                    continue
                med = float(np.median(s))
                q75 = float(np.percentile(s, 75))
                q25 = float(np.percentile(s, 25))
                iqr = max(1e-12, q75 - q25)
                thr = med + 3.0 * iqr
                count = int((s > thr).sum())
                items.append({"col": c, "median": med, "iqr": float(iqr), "thr": float(thr), "anomaly_count": count})
            return items
        out["wavelet"] = _anomaly_counts(df[wave_cols]) if wave_cols else []
        out["log"] = _anomaly_counts(df[log_cols]) if log_cols else []
        return out
    if ct_df is not None:
        wave_log_std["ct"] = _wave_log_std_anomaly(ct_df)
    if dose_df is not None:
        wave_log_std["dose"] = _wave_log_std_anomaly(dose_df)
    rep["wavelet_log_sigma_std_anomaly"] = wave_log_std
    wave_log_feat_std = {}
    def _column_std_stats(df, cols):
        if not cols:
            return {"median": None, "iqr": None, "thr": None, "count_above_thr": 0, "top10": []}
        vals = df[cols].apply(pd.to_numeric, errors="coerce")
        s = vals.std(axis=0, ddof=1).dropna()
        if s.size == 0:
            return {"median": None, "iqr": None, "thr": None, "count_above_thr": 0, "top10": []}
        med = float(np.median(s.values))
        q75 = float(np.percentile(s.values, 75))
        q25 = float(np.percentile(s.values, 25))
        iqr = max(1e-12, q75 - q25)
        thr = med + 3.0 * iqr
        cnt = int((s.values > thr).sum())
        top_idx = np.argsort(s.values)[::-1][:10]
        top = [{"feature": cols[i], "std": float(s.values[i])} for i in top_idx]
        return {"median": med, "iqr": float(iqr), "thr": float(thr), "count_above_thr": cnt, "top10": top}
    if ct_df is not None:
        ct_cols_wave = [c for c in ct_df.columns if "wavelet" in c.lower()]
        ct_cols_log = [c for c in ct_df.columns if ("log-sigma" in c.lower() or ("log" in c.lower() and "original" not in c.lower()))]
        wave_log_feat_std["ct_wavelet_std_summary"] = _column_std_stats(ct_df, ct_cols_wave)
        wave_log_feat_std["ct_log_sigma_std_summary"] = _column_std_stats(ct_df, ct_cols_log)
    if dose_df is not None:
        dose_cols_wave = [c for c in dose_df.columns if "wavelet" in c.lower()]
        dose_cols_log = [c for c in dose_df.columns if ("log-sigma" in c.lower() or ("log" in c.lower() and "original" not in c.lower()))]
        wave_log_feat_std["dose_wavelet_std_summary"] = _column_std_stats(dose_df, dose_cols_wave)
        wave_log_feat_std["dose_log_sigma_std_summary"] = _column_std_stats(dose_df, dose_cols_log)
    rep["wavelet_log_sigma_feature_std"] = wave_log_feat_std
    samples = []
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path)
        idx_valid = idx[(~idx["missing_ct"]) & (~idx["missing_dose"])]
        if idx_valid.shape[0] > 0:
            smp = idx_valid.sample(n=min(int(n_sample), int(idx_valid.shape[0])), random_state=42)
            for _, rec in smp.iterrows():
                cid = rec["case_id"]
                ct_img = sitk.ReadImage(rec["ct_path"])
                dose_img = sitk.ReadImage(rec["dose_path"])
                ct_arr_raw = sitk.GetArrayFromImage(ct_img).astype(np.float32)
                dose_arr_raw = sitk.GetArrayFromImage(dose_img).astype(np.float32)
                ct_proc = ct_intensity_process(ct_img)
                dose_proc = dose_intensity_process(dose_img)
                ct_arr_proc = sitk.GetArrayFromImage(ct_proc).astype(np.float32)
                dose_arr_proc = sitk.GetArrayFromImage(dose_proc).astype(np.float32)
                samples.append({
                    "case_id": cid,
                    "ct_raw_min": float(np.min(ct_arr_raw)),
                    "ct_raw_max": float(np.max(ct_arr_raw)),
                    "ct_raw_mean": float(np.mean(ct_arr_raw)),
                    "ct_proc_min": float(np.min(ct_arr_proc)),
                    "ct_proc_max": float(np.max(ct_arr_proc)),
                    "ct_proc_mean": float(np.mean(ct_arr_proc)),
                    "dose_raw_min": float(np.min(dose_arr_raw)),
                    "dose_raw_max": float(np.max(dose_arr_raw)),
                    "dose_raw_mean": float(np.mean(dose_arr_raw)),
                    "dose_proc_min": float(np.min(dose_arr_proc)),
                    "dose_proc_max": float(np.max(dose_arr_proc)),
                    "dose_proc_mean": float(np.mean(dose_arr_proc)),
                })
    rep["sample_cases"] = samples
    with open(os.path.join(feat_dir, "qc_report.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    if samples:
        pd.DataFrame(samples).to_csv(os.path.join(feat_dir, "qc_sample_cases.csv"), index=False)
    if ct_df is not None and dose_df is not None:
        def _get_std_col(df, organ, kind):
            candidates = [
                f"{organ}__sym__mean__{kind}__original_firstorder_StandardDeviation",
                f"{organ}_L__{kind}__original_firstorder_StandardDeviation",
                f"{organ}_R__{kind}__original_firstorder_StandardDeviation",
            ]
            for c in candidates:
                if c in df.columns:
                    return c
            for c in df.columns:
                if c.startswith(organ) and f"__{kind}__original_firstorder_StandardDeviation" in c:
                    return c
            return None
        rows = []
        for _, r in ct_df[["case_id"]].iterrows():
            cid = r["case_id"]
            rows.append({"case_id": cid})
        out_df = pd.DataFrame(rows)
        for organ in ["parotid", "submand"]:
            c_ct = _get_std_col(ct_df, organ, "ct")
            c_dose = _get_std_col(dose_df, organ, "dose")
            if c_ct is not None:
                out_df[f"{organ}_ct_std"] = _safe_float_series(ct_df[c_ct])
            if c_dose is not None:
                out_df[f"{organ}_dose_std"] = _safe_float_series(dose_df[c_dose])
        out_df.to_csv(os.path.join(feat_dir, "qc_std_ct_vs_dose.csv"), index=False)


def _read_image_info(path):
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return {
        "size": tuple(int(x) for x in r.GetSize()),
        "spacing": tuple(float(x) for x in r.GetSpacing()),
        "origin": tuple(float(x) for x in r.GetOrigin()),
        "direction": tuple(float(x) for x in r.GetDirection()),
    }


def _compare_geom(a, b):
    diffs = []
    if a["size"] != b["size"]:
        diffs.append(f"size:{a['size']}!={b['size']}")
    if not arrays_close(a["spacing"], b["spacing"]):
        diffs.append(f"spacing:{a['spacing']}!={b['spacing']}")
    if not arrays_close(a["origin"], b["origin"]):
        diffs.append(f"origin:{a['origin']}!={b['origin']}")
    if not arrays_close(a["direction"], b["direction"]):
        diffs.append("direction_mismatch")
    return diffs


def _read_plan_prescription(case_id):
    p = os.path.join(PLAN_JSON_ROOT, f"{case_id}_convert", f"{case_id}_RP_plan.json")
    if not os.path.exists(p):
        return {"plan_json_path": p, "TargetPrescriptionDose": None}
    try:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        pres = j.get("Prescription") or {}
        return {
            "plan_json_path": p,
            "TargetPrescriptionDose": pres.get("TargetPrescriptionDose"),
            "NumberOfFractionsPlanned": pres.get("NumberOfFractionsPlanned"),
            "DosePerFraction": pres.get("DosePerFraction"),
        }
    except Exception:
        return {"plan_json_path": p, "TargetPrescriptionDose": None}


def _dose_scale_factor(dose_max):
    try:
        mx = float(dose_max)
    except Exception:
        return 1.0
    return 0.01 if mx > 200.0 else 1.0


def _numeric_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"min": None, "p1": None, "p5": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(a)),
        "p1": float(np.percentile(a, 1)),
        "p5": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def _nan_inf_report(df):
    cols = [c for c in df.columns if c != "case_id"]
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    v = vals.values
    out = {
        "total_nan": int(vals.isna().sum().sum()),
        "total_inf": int(np.isinf(v).sum()),
    }
    wave_cols = [c for c in cols if "wavelet" in c.lower()]
    log_cols = [c for c in cols if ("log-sigma" in c.lower() or ("log" in c.lower() and "original" not in c.lower()))]
    if wave_cols:
        wv = df[wave_cols].apply(pd.to_numeric, errors="coerce").values
        out["wavelet_nan"] = int(np.isnan(wv).sum())
        out["wavelet_inf"] = int(np.isinf(wv).sum())
    else:
        out["wavelet_nan"] = 0
        out["wavelet_inf"] = 0
    if log_cols:
        lv = df[log_cols].apply(pd.to_numeric, errors="coerce").values
        out["log_sigma_nan"] = int(np.isnan(lv).sum())
        out["log_sigma_inf"] = int(np.isinf(lv).sum())
    else:
        out["log_sigma_nan"] = 0
        out["log_sigma_inf"] = 0
    return out


def run_task12_qc(n_repro=5, voxel_small_thresholds=(50, 100), random_state=42):
    feat_dir = os.path.join(OUTPUT_ROOT, "features")
    cur_dir = os.path.join(OUTPUT_ROOT, "curation")
    ensure_dir(feat_dir)
    idx_path = os.path.join(cur_dir, "case_file_index.csv")
    if not os.path.exists(idx_path):
        raise RuntimeError(f"missing case index: {idx_path}")
    idx = pd.read_csv(idx_path)
    idx = idx[(~idx["missing_ct"]) & (~idx["missing_dose"]) & (~idx["missing_parotid_L_mask"]) & (~idx["missing_parotid_R_mask"]) & (~idx["missing_submand_L_mask"]) & (~idx["missing_submand_R_mask"])]

    geom_rows = []
    for _, r in idx.iterrows():
        cid = r["case_id"]
        ct_info = _read_image_info(r["ct_path"])
        dose_info = _read_image_info(r["dose_path"])
        pl_info = _read_image_info(r["parotid_L_mask_path"])
        pr_info = _read_image_info(r["parotid_R_mask_path"])
        sl_info = _read_image_info(r["submand_L_mask_path"])
        sr_info = _read_image_info(r["submand_R_mask_path"])

        diffs = []
        diffs += [f"ct_vs_dose:{d}" for d in _compare_geom(ct_info, dose_info)]
        diffs += [f"ct_vs_parotid_L:{d}" for d in _compare_geom(ct_info, pl_info)]
        diffs += [f"ct_vs_parotid_R:{d}" for d in _compare_geom(ct_info, pr_info)]
        diffs += [f"ct_vs_submand_L:{d}" for d in _compare_geom(ct_info, sl_info)]
        diffs += [f"ct_vs_submand_R:{d}" for d in _compare_geom(ct_info, sr_info)]

        geom_rows.append({
            "case_id": cid,
            "ct_path": r["ct_path"],
            "dose_path": r["dose_path"],
            "parotid_L_mask_path": r["parotid_L_mask_path"],
            "parotid_R_mask_path": r["parotid_R_mask_path"],
            "submand_L_mask_path": r["submand_L_mask_path"],
            "submand_R_mask_path": r["submand_R_mask_path"],
            "geom_match": int(len(diffs) == 0),
            "diffs": ";".join(diffs),
        })
    geom_df = pd.DataFrame(geom_rows)
    geom_df.to_csv(os.path.join(feat_dir, "qc_geom.csv"), index=False)

    ct_raw_path = os.path.join(feat_dir, "ct_radiomics_raw.csv")
    dose_raw_path = os.path.join(feat_dir, "dosiomics_raw.csv")
    dvh_raw_path = os.path.join(feat_dir, "dvh_raw.csv")

    ct_df = pd.read_csv(ct_raw_path) if os.path.exists(ct_raw_path) else None
    dose_df = pd.read_csv(dose_raw_path) if os.path.exists(dose_raw_path) else None
    dvh_df = pd.read_csv(dvh_raw_path) if os.path.exists(dvh_raw_path) else None

    if ct_df is None or dose_df is None:
        raise RuntimeError("missing radiomics raw tables in outputs/features")

    voxel_cols = {
        "parotid_L": "parotid_L__ct__diagnostics_Mask-original_VoxelNum",
        "parotid_R": "parotid_R__ct__diagnostics_Mask-original_VoxelNum",
        "submand_L": "submand_L__ct__diagnostics_Mask-original_VoxelNum",
        "submand_R": "submand_R__ct__diagnostics_Mask-original_VoxelNum",
    }
    vol_cols = {
        "parotid_L": "parotid_L__ct__original_shape_MeshVolume",
        "parotid_R": "parotid_R__ct__original_shape_MeshVolume",
        "submand_L": "submand_L__ct__original_shape_MeshVolume",
        "submand_R": "submand_R__ct__original_shape_MeshVolume",
    }

    roi_rows = []
    for organ, c in voxel_cols.items():
        if c not in ct_df.columns:
            raise RuntimeError(f"missing voxel column: {c}")
    for organ, c in vol_cols.items():
        if c not in ct_df.columns:
            raise RuntimeError(f"missing volume column: {c}")

    for _, r in ct_df[["case_id"] + list(voxel_cols.values()) + list(vol_cols.values())].iterrows():
        row = {"case_id": r["case_id"]}
        for organ, c in voxel_cols.items():
            row[f"{organ}_voxelnum"] = float(r[c]) if pd.notna(r[c]) else np.nan
        for organ, c in vol_cols.items():
            row[f"{organ}_meshvol_mm3"] = float(r[c]) if pd.notna(r[c]) else np.nan
        roi_rows.append(row)

    roi_df = pd.DataFrame(roi_rows)
    for thr in voxel_small_thresholds:
        for organ in voxel_cols.keys():
            roi_df[f"{organ}_voxelnum_lt_{thr}"] = (pd.to_numeric(roi_df[f"{organ}_voxelnum"], errors="coerce") < float(thr)).astype(int)
    roi_df.to_csv(os.path.join(feat_dir, "qc_roi_voxel_volume.csv"), index=False)

    roi_stats = {"voxelnum_thresholds": list(voxel_small_thresholds), "voxelnum": {}, "meshvol_mm3": {}}
    for organ in voxel_cols.keys():
        roi_stats["voxelnum"][organ] = _numeric_stats(pd.to_numeric(roi_df[f"{organ}_voxelnum"], errors="coerce"))
        roi_stats["meshvol_mm3"][organ] = _numeric_stats(pd.to_numeric(roi_df[f"{organ}_meshvol_mm3"], errors="coerce"))

    plot_dir = os.path.join(feat_dir, "qc_plots")
    ensure_dir(plot_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extremes = []
    for organ in voxel_cols.keys():
        vol = pd.to_numeric(roi_df[f"{organ}_meshvol_mm3"], errors="coerce").dropna()
        if vol.size > 0:
            plt.figure(figsize=(7, 4))
            plt.hist(vol.values, bins=50)
            plt.title(f"{organ} MeshVolume (mm3) histogram")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{organ}_meshvolume_hist.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(7, 2.6))
            plt.boxplot(vol.values, vert=False, showfliers=True)
            plt.title(f"{organ} MeshVolume (mm3) boxplot")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{organ}_meshvolume_box.png"), dpi=150)
            plt.close()

        dfv = roi_df[["case_id", f"{organ}_voxelnum", f"{organ}_meshvol_mm3"]].copy()
        dfv["volume"] = pd.to_numeric(dfv[f"{organ}_meshvol_mm3"], errors="coerce")
        dfv["voxelnum"] = pd.to_numeric(dfv[f"{organ}_voxelnum"], errors="coerce")
        dfv = dfv.dropna(subset=["volume"])
        if dfv.shape[0] > 0:
            dfv_small = dfv.sort_values("volume", ascending=True).head(5)
            dfv_large = dfv.sort_values("volume", ascending=False).head(5)
            for rank, (_, rr) in enumerate(dfv_small.iterrows(), start=1):
                extremes.append({"organ": organ, "kind": "small", "rank": int(rank), "case_id": rr["case_id"], "meshvol_mm3": float(rr["volume"]), "voxelnum": float(rr["voxelnum"]) if pd.notna(rr["voxelnum"]) else None})
            for rank, (_, rr) in enumerate(dfv_large.iterrows(), start=1):
                extremes.append({"organ": organ, "kind": "large", "rank": int(rank), "case_id": rr["case_id"], "meshvol_mm3": float(rr["volume"]), "voxelnum": float(rr["voxelnum"]) if pd.notna(rr["voxelnum"]) else None})
    if extremes:
        pd.DataFrame(extremes).to_csv(os.path.join(feat_dir, "qc_roi_volume_extremes.csv"), index=False)

    sample_ids = set(idx.sample(n=min(20, int(idx.shape[0])), random_state=int(random_state))["case_id"].astype(str).tolist())
    if extremes:
        for x in extremes:
            sample_ids.add(str(x["case_id"]))
    geom_sample_df = geom_df[geom_df["case_id"].astype(str).isin(sample_ids)].copy()
    geom_sample_df.to_csv(os.path.join(feat_dir, "qc_geom_sample.csv"), index=False)

    def _img_info(img):
        return {
            "size": tuple(int(x) for x in img.GetSize()),
            "spacing": tuple(float(x) for x in img.GetSpacing()),
            "origin": tuple(float(x) for x in img.GetOrigin()),
            "direction": tuple(float(x) for x in img.GetDirection()),
        }

    def _needs_align_to_ct(ct_img, img):
        return (
            (tuple(int(x) for x in img.GetSize()) != tuple(int(x) for x in ct_img.GetSize()))
            or (not arrays_close(ct_img.GetSpacing(), img.GetSpacing()))
            or (not arrays_close(ct_img.GetOrigin(), img.GetOrigin()))
            or (not arrays_close(ct_img.GetDirection(), img.GetDirection()))
        )

    def _align_to_ct(ct_img, img, *, is_mask):
        if _needs_align_to_ct(ct_img, img):
            return resample_to_reference(img, ct_img, is_mask=bool(is_mask))
        return img

    aligned_rows = []
    sample_idx = idx[idx["case_id"].astype(str).isin(sample_ids)].copy()
    for _, r in sample_idx.iterrows():
        cid = str(r["case_id"])
        try:
            ct_img = sitk.ReadImage(r["ct_path"])
            ct_i = _img_info(ct_img)
            dose_img = _align_to_ct(ct_img, sitk.ReadImage(r["dose_path"]), is_mask=False)
            pl_img = _align_to_ct(ct_img, sitk.ReadImage(r["parotid_L_mask_path"]), is_mask=True)
            pr_img = _align_to_ct(ct_img, sitk.ReadImage(r["parotid_R_mask_path"]), is_mask=True)
            sl_img = _align_to_ct(ct_img, sitk.ReadImage(r["submand_L_mask_path"]), is_mask=True)
            sr_img = _align_to_ct(ct_img, sitk.ReadImage(r["submand_R_mask_path"]), is_mask=True)

            dose_i = _img_info(dose_img)
            pl_i = _img_info(pl_img)
            pr_i = _img_info(pr_img)
            sl_i = _img_info(sl_img)
            sr_i = _img_info(sr_img)

            diffs2 = []
            diffs2 += [f"ct_vs_dose:{d}" for d in _compare_geom(ct_i, dose_i)]
            diffs2 += [f"ct_vs_parotid_L:{d}" for d in _compare_geom(ct_i, pl_i)]
            diffs2 += [f"ct_vs_parotid_R:{d}" for d in _compare_geom(ct_i, pr_i)]
            diffs2 += [f"ct_vs_submand_L:{d}" for d in _compare_geom(ct_i, sl_i)]
            diffs2 += [f"ct_vs_submand_R:{d}" for d in _compare_geom(ct_i, sr_i)]

            aligned_rows.append({"case_id": cid, "aligned_geom_match": int(len(diffs2) == 0), "diffs": ";".join(diffs2)})
        except Exception as e:
            aligned_rows.append({"case_id": cid, "aligned_geom_match": 0, "diffs": f"error:{e}"})

    aligned_df = pd.DataFrame(aligned_rows)
    aligned_df.to_csv(os.path.join(feat_dir, "qc_geom_aligned_sample.csv"), index=False)

    hu_cols = [c for c in ct_df.columns if "__ct__original_firstorder_" in c and "__sym__" not in c]
    dose_max_cols = [c for c in dose_df.columns if "__dose__original_firstorder_Maximum" in c and "__sym__" not in c]
    hu_vals = {}
    for k in ["Minimum", "Maximum", "Mean"]:
        cols = [c for c in hu_cols if f"__ct__original_firstorder_{k}" in c]
        vals = []
        for c in cols:
            vals.extend(pd.to_numeric(ct_df[c], errors="coerce").dropna().tolist())
        hu_vals[k] = _numeric_stats(vals)

    dose_roi_max_vals = []
    for c in dose_max_cols:
        dose_roi_max_vals.extend(pd.to_numeric(dose_df[c], errors="coerce").dropna().tolist())
    dose_roi_max_stats = _numeric_stats(dose_roi_max_vals)

    nan_inf = {
        "ct_radiomics_raw": _nan_inf_report(ct_df),
        "dosiomics_raw": _nan_inf_report(dose_df),
    }
    if dvh_df is not None:
        nan_inf["dvh_raw"] = _nan_inf_report(dvh_df)

    dose_rx_rows = []
    for _, r in idx.iterrows():
        cid = r["case_id"]
        pres = _read_plan_prescription(cid)
        rx = pres.get("TargetPrescriptionDose")
        try:
            dose_img = sitk.ReadImage(r["dose_path"])
            dose_arr = sitk.GetArrayFromImage(dose_img).astype(np.float32)
            raw_min = float(np.min(dose_arr))
            raw_max = float(np.max(dose_arr))
            raw_mean = float(np.mean(dose_arr))
            sf = _dose_scale_factor(raw_max)
            dose_arr = dose_arr * float(sf)
            dose_arr[dose_arr < 0.0] = 0.0
            gy_min = float(np.min(dose_arr))
            gy_max = float(np.max(dose_arr))
            gy_mean = float(np.mean(dose_arr))
            neg_raw = int((sitk.GetArrayFromImage(dose_img) < 0).sum())
        except Exception:
            raw_min = None
            raw_max = None
            raw_mean = None
            sf = None
            gy_min = None
            gy_max = None
            gy_mean = None
            neg_raw = None

        ratio = None
        diff = None
        if rx is not None and gy_max is not None:
            try:
                ratio = float(gy_max) / float(rx) if float(rx) != 0.0 else None
                diff = float(gy_max) - float(rx)
            except Exception:
                ratio = None
                diff = None

        dose_rx_rows.append({
            "case_id": cid,
            "dose_path": r["dose_path"],
            "plan_json_path": pres.get("plan_json_path"),
            "rx_gy": rx,
            "fractions": pres.get("NumberOfFractionsPlanned"),
            "dose_per_fraction": pres.get("DosePerFraction"),
            "dose_raw_min": raw_min,
            "dose_raw_max": raw_max,
            "dose_raw_mean": raw_mean,
            "dose_scale_factor": sf,
            "dose_gy_min": gy_min,
            "dose_gy_max": gy_max,
            "dose_gy_mean": gy_mean,
            "dose_gy_max_minus_rx": diff,
            "dose_gy_max_div_rx": ratio,
            "dose_raw_neg_voxels": neg_raw,
        })
    dose_rx_df = pd.DataFrame(dose_rx_rows)
    dose_rx_df.to_csv(os.path.join(feat_dir, "qc_dose_vs_prescription.csv"), index=False)
    anomalies = dose_rx_df.copy()
    rx = pd.to_numeric(anomalies["rx_gy"], errors="coerce")
    gy_max = pd.to_numeric(anomalies["dose_gy_max"], errors="coerce")
    gy_min = pd.to_numeric(anomalies["dose_gy_min"], errors="coerce")
    ratio = pd.to_numeric(anomalies["dose_gy_max_div_rx"], errors="coerce")
    diff = pd.to_numeric(anomalies["dose_gy_max_minus_rx"], errors="coerce")
    neg = pd.to_numeric(anomalies["dose_raw_neg_voxels"], errors="coerce").fillna(0)
    flag = (
        rx.isna()
        | gy_max.isna()
        | (neg > 0)
        | (gy_min < 0)
        | (ratio < 0.7)
        | (ratio > 1.5)
        | (diff.abs() > 15.0)
    )
    anomalies = anomalies[flag]
    anomalies.to_csv(os.path.join(feat_dir, "qc_dose_vs_prescription_anomalies.csv"), index=False)

    params_snapshot = {
        "target_spacing_mm": [1.0, 1.0, 1.0],
        "ct_clip_hu": [-1000.0, 1000.0],
        "dose_scale_rule": "scale=0.01 if max>200 else 1.0; clip<0 to 0",
        "mask_resample_interpolator": "nearest",
        "image_resample_interpolator": "linear",
        "largest_connected_component": True,
        "image_types": {"Original": {}, "LoG": {"sigma": [1.0, 2.0, 3.0]}, "Wavelet": {}},
        "bin_width_strategy": "per-organ ROI range / 128 (min 1e-4) else 1.0",
        "pyradiomics": {"normalize": False, "correctMask": True},
    }
    with open(os.path.join(feat_dir, "params_snapshot.json"), "w") as f:
        json.dump(params_snapshot, f, ensure_ascii=False, indent=2)

    repro_rows = []
    smp = idx.sample(n=min(int(n_repro), int(idx.shape[0])), random_state=int(random_state))
    for _, r in smp.iterrows():
        cid = r["case_id"]
        try:
            ct = sitk.ReadImage(r["ct_path"])
            dose = sitk.ReadImage(r["dose_path"])
            pl = sitk.ReadImage(r["parotid_L_mask_path"])
            pr = sitk.ReadImage(r["parotid_R_mask_path"])
            sl = sitk.ReadImage(r["submand_L_mask_path"])
            sr = sitk.ReadImage(r["submand_R_mask_path"])
            if (not arrays_close(ct.GetSpacing(), dose.GetSpacing()) or
                not arrays_close(ct.GetOrigin(), dose.GetOrigin()) or
                not arrays_close(ct.GetDirection(), dose.GetDirection())):
                dose = resample_to_reference(dose, ct, is_mask=False)
            ct_i = resample_to_isotropic(ct, spacing=(1.0, 1.0, 1.0), is_mask=False)
            dose_i = resample_to_isotropic(dose, spacing=(1.0, 1.0, 1.0), is_mask=False)
            ct_i = ct_intensity_process(ct_i)
            dose_i = dose_intensity_process(dose_i)
            a1 = {}
            a1.update(extract_for_one_mask(ct, dose, ct_i, dose_i, pl, "parotid_L", (1.0, 1.0, 1.0)))
            a1.update(extract_for_one_mask(ct, dose, ct_i, dose_i, pr, "parotid_R", (1.0, 1.0, 1.0)))
            a1.update(extract_for_one_mask(ct, dose, ct_i, dose_i, sl, "submand_L", (1.0, 1.0, 1.0)))
            a1.update(extract_for_one_mask(ct, dose, ct_i, dose_i, sr, "submand_R", (1.0, 1.0, 1.0)))
            a2 = {}
            a2.update(extract_for_one_mask(ct, dose, ct_i, dose_i, pl, "parotid_L", (1.0, 1.0, 1.0)))
            a2.update(extract_for_one_mask(ct, dose, ct_i, dose_i, pr, "parotid_R", (1.0, 1.0, 1.0)))
            a2.update(extract_for_one_mask(ct, dose, ct_i, dose_i, sl, "submand_L", (1.0, 1.0, 1.0)))
            a2.update(extract_for_one_mask(ct, dose, ct_i, dose_i, sr, "submand_R", (1.0, 1.0, 1.0)))
            keys = sorted(set(a1.keys()) | set(a2.keys()))
            max_abs = 0.0
            max_rel = 0.0
            bad = 0
            for k in keys:
                v1 = a1.get(k)
                v2 = a2.get(k)
                try:
                    f1 = float(v1)
                    f2 = float(v2)
                except Exception:
                    continue
                d = abs(f1 - f2)
                if d > max_abs:
                    max_abs = d
                denom = max(abs(f1), abs(f2), 1e-12)
                rdiff = d / denom
                if rdiff > max_rel:
                    max_rel = rdiff
                if d > 1e-8:
                    bad += 1
            repro_rows.append({"case_id": cid, "max_abs_diff": float(max_abs), "max_rel_diff": float(max_rel), "n_numeric_diff_gt_1e-8": int(bad)})
        except Exception as e:
            repro_rows.append({"case_id": cid, "error": str(e)})

    repro_df = pd.DataFrame(repro_rows)
    repro_df.to_csv(os.path.join(feat_dir, "qc_reproducibility.csv"), index=False)

    rep = {
        "geom": {
            "total": int(geom_df.shape[0]),
            "match": int((geom_df["geom_match"] == 1).sum()),
            "mismatch": int((geom_df["geom_match"] == 0).sum()),
        },
        "geom_aligned_sample": {
            "total": int(aligned_df.shape[0]),
            "match": int((aligned_df["aligned_geom_match"] == 1).sum()) if "aligned_geom_match" in aligned_df.columns else 0,
            "mismatch": int((aligned_df["aligned_geom_match"] == 0).sum()) if "aligned_geom_match" in aligned_df.columns else int(aligned_df.shape[0]),
            "error": int(aligned_df["diffs"].astype(str).str.startswith("error:").sum()) if "diffs" in aligned_df.columns else 0,
        },
        "roi_stats": roi_stats,
        "ct_hu_stats": hu_vals,
        "dose_roi_max_stats": dose_roi_max_stats,
        "nan_inf": nan_inf,
    }
    with open(os.path.join(feat_dir, "qc_task12_report.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    return rep

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--augment-sym", action="store_true")
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--qc-task12", action="store_true")
    ap.add_argument("--save-cropped", action="store_true", help="保存裁剪后的CT和Dose文件")
    ap.add_argument("--ct-crop-dir", type=str, default=None, help="裁剪CT文件输出目录")
    ap.add_argument("--dose-crop-dir", type=str, default=None, help="裁剪Dose文件输出目录")
    args = ap.parse_args()
    if args.prepare_only:
        prepare_ml_tables()
    elif args.augment_sym:
        augment_symmetric_features_in_features_dir()
        prepare_ml_tables()
    elif args.qc:
        run_qc_checks(n_sample=5)
    elif args.qc_task12:
        run_task12_qc()
    else:
        run(limit=args.limit, save_cropped=args.save_cropped,
            ct_crop_dir=args.ct_crop_dir, dose_crop_dir=args.dose_crop_dir)
        prepare_ml_tables()

"""机器学习示例：NCC 粗定位 + HOG 特征 + SVM 验证，剔除模板匹配的误匹配。

传统 NCC 在背景相似、遮挡、光照变化下会产生假峰。
思路：用 NCC 找出所有候选位置（低阈值），对每个候选窗口提取 HOG 特征，
训练好的 SVM 判别"这是不是真正的模板目标"，从而提高查准率。

依赖: pip install -r requirements-ml.txt
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    import sklearn  # noqa: F401
except ImportError as e:
    raise ImportError("需要 scikit-learn：pip install -r requirements-ml.txt") from e


def hog_features(patch: np.ndarray, win_size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """提取 HOG 特征（patch 会被缩放到 win_size，w×h）。"""
    patch = cv2.resize(np.asarray(patch, dtype=np.uint8), win_size)
    hog = cv2.HOGDescriptor(
        win_size,            # 窗口
        (16, 16),            # block
        (8, 8),              # block stride
        (8, 8),              # cell
        9,                   # bins
    )
    return hog.compute(patch).ravel().astype(np.float32)


def make_training_set(template: np.ndarray, n_pos: int = 60, n_neg: int = 120,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """构造训练集。

    正样本：模板的扰动版本（亮度/对比度/小噪声）。
    负样本：随机纹理块（可用真实产线上的误匹配窗口替换）。
    """
    rng = np.random.default_rng(seed)
    h, w = template.shape
    pos = []
    for _ in range(n_pos):
        alpha = rng.uniform(0.7, 1.3)      # 对比度
        beta = rng.uniform(-20, 20)        # 亮度
        p = template * alpha + beta + rng.normal(0, 5, template.shape)
        pos.append(hog_features(p))
    neg = []
    for _ in range(n_neg):
        tex = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 2.0)
        neg.append(hog_features(tex))
    X = np.stack(pos + neg)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    return X, y


def verify_candidates(
    image: np.ndarray,
    template: np.ndarray,
    candidates: list[dict],
    seed: int = 0,
) -> list[dict]:
    """对 NCC 候选结果做 SVM 验证，返回带 `svm_score` 的结果列表（降序）。"""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X, y = make_training_set(template, seed=seed)
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True))
    clf.fit(X, y)

    out = []
    for c in candidates:
        x, yy = int(round(c["x"])), int(round(c["y"]))
        patch = image[yy:yy + template.shape[0], x:x + template.shape[1]]
        if patch.shape != template.shape:
            continue
        feat = hog_features(patch)[None, :]
        out.append({**c, "svm_score": float(clf.predict_proba(feat)[0, 1])})
    return sorted(out, key=lambda d: -d["svm_score"])


if __name__ == "__main__":
    from data.synthetic import make_template_scene
    from traditional.template_matching.ncc_matching import find_best_matches

    img, tpl, gt_xy = make_template_scene(noise=5.0)
    cands = find_best_matches(img, tpl, min_score=0.5, nms_radius=4, max_count=20)
    print(f"NCC 低阈值候选数: {len(cands)}")
    verified = verify_candidates(img, tpl, cands)
    best = verified[0]
    err = np.hypot(best["x"] - gt_xy[0], best["y"] - gt_xy[1])
    print(f"SVM 验证后 Top-1: ncc={best['score']:.3f} svm={best['svm_score']:.3f} "
          f"误差={err:.3f} px")

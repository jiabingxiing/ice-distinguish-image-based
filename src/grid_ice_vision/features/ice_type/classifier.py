"""推理分类器抽象与可导出 PyTorch 头（支持 Linear/RBF SVM、LR、MLP 导出单一 ONNX）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class LinearSvmHead(nn.Module):
    """将 StandardScaler + 线性分类器（LinearSVC / LogisticRegression）封装为 PyTorch 张量算子。"""

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        safe_scale = np.where(np.asarray(scale) == 0.0, 1.0, scale)
        self.register_buffer("scale", torch.as_tensor(safe_scale, dtype=torch.float32))
        self.linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
            self.linear.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        z = (x - self.get_buffer("mean")) / self.get_buffer("scale")
        return self.linear(z)


class RbfSvmTorchHead(nn.Module):
    """将 StandardScaler + sklearn 多类 RBF SVC 转为纯 PyTorch 张量算子（可导出 ONNX）。

    计算精确的 One-vs-One 决策面，并通过每类对抗其余类的平均边际构造 C 维 logits，
    配合温度缩放后与线性头/微调头共享同一概率校准与拒识接口。
    """

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        support_vectors: np.ndarray,
        dual_coef: np.ndarray,
        intercept: np.ndarray,
        n_support: np.ndarray,
        gamma: float,
    ) -> None:
        super().__init__()
        self.num_classes = len(n_support)
        self.gamma = float(gamma)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        safe_scale = np.where(np.asarray(scale) == 0.0, 1.0, scale)
        self.register_buffer("scale", torch.as_tensor(safe_scale, dtype=torch.float32))

        sv = torch.as_tensor(support_vectors, dtype=torch.float32)
        self.register_buffer("sv", sv)
        self.register_buffer("sv_norm2", (sv * sv).sum(dim=1, keepdim=True).t())
        self.register_buffer("dual_coef", torch.as_tensor(dual_coef, dtype=torch.float32))
        self.register_buffer("intercept", torch.as_tensor(intercept, dtype=torch.float32))

        starts: list[int] = []
        ends: list[int] = []
        cur = 0
        for cnt in n_support:
            starts.append(cur)
            cur += int(cnt)
            ends.append(cur)
        self._starts = tuple(starts)
        self._ends = tuple(ends)

    def ovo_decision(self, x: Tensor) -> Tensor:
        """返回与 sklearn `SVC.decision_function` 一致的 (B, C*(C-1)//2) 成对决策值。"""
        z = (x - self.get_buffer("mean")) / self.get_buffer("scale")
        sv = self.get_buffer("sv")
        sv_norm2 = self.get_buffer("sv_norm2")
        dual_coef = self.get_buffer("dual_coef")
        intercept = self.get_buffer("intercept")

        z_norm2 = (z * z).sum(dim=1, keepdim=True)
        dist2 = (z_norm2 + sv_norm2 - 2.0 * (z @ sv.t())).clamp_min(0.0)
        K = torch.exp(-self.gamma * dist2)

        pair_vals: list[Tensor] = []
        p = 0
        c = self.num_classes
        for i in range(c):
            si, ei = self._starts[i], self._ends[i]
            for j in range(i + 1, c):
                sj, ej = self._starts[j], self._ends[j]
                coef_i = dual_coef[j - 1, si:ei]
                coef_j = dual_coef[i, sj:ej]
                val = K[:, si:ei] @ coef_i + K[:, sj:ej] @ coef_j + intercept[p]
                pair_vals.append(val)
                p += 1
        return torch.stack(pair_vals, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        ovo = self.ovo_decision(x)
        c = self.num_classes
        scores = [x.new_zeros((x.shape[0],)) for _ in range(c)]
        p = 0
        for i in range(c):
            for j in range(i + 1, c):
                d = ovo[:, p]
                scores[i] = scores[i] + d
                scores[j] = scores[j] - d
                p += 1
        return torch.stack(scores, dim=1) / max(c - 1, 1)


class MlpTorchHead(nn.Module):
    """将 StandardScaler + sklearn MLPClassifier 转为纯 PyTorch MLP。"""

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        coefs: list[np.ndarray],
        intercepts: list[np.ndarray],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        safe_scale = np.where(np.asarray(scale) == 0.0, 1.0, scale)
        self.register_buffer("scale", torch.as_tensor(safe_scale, dtype=torch.float32))
        layers: list[nn.Module] = []
        for idx, (w, b) in enumerate(zip(coefs, intercepts, strict=True)):
            lin = nn.Linear(w.shape[0], w.shape[1])
            with torch.no_grad():
                lin.weight.copy_(torch.as_tensor(w.T, dtype=torch.float32))
                lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
            layers.append(lin)
            if idx < len(coefs) - 1:
                layers.append(nn.Tanh() if activation == "tanh" else nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        z = (x - self.get_buffer("mean")) / self.get_buffer("scale")
        return self.net(z)


class IceTypeLinearHead(nn.Module):
    """P3 末端微调四分类线性头。"""

    def __init__(self, in_dim: int = 576, num_classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.drop(x))


def export_torch_head(scaler: Any, classifier: Any) -> nn.Module:
    """将已拟合的 sklearn StandardScaler + 分类器转为纯 PyTorch 可导出模块。"""
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    cls_name = type(classifier).__name__

    if cls_name == "SVC" and getattr(classifier, "kernel", None) == "rbf":
        gamma = float(classifier._gamma)
        return RbfSvmTorchHead(
            mean=mean,
            scale=scale,
            support_vectors=classifier.support_vectors_,
            dual_coef=classifier.dual_coef_,
            intercept=classifier.intercept_,
            n_support=classifier.n_support_,
            gamma=gamma,
        )
    if cls_name in ("LinearSVC", "LogisticRegression"):
        return LinearSvmHead(
            mean=mean,
            scale=scale,
            weight=classifier.coef_,
            bias=classifier.intercept_,
        )
    if cls_name == "MLPClassifier":
        return MlpTorchHead(
            mean=mean,
            scale=scale,
            coefs=list(classifier.coefs_),
            intercepts=list(classifier.intercepts_),
            activation=str(getattr(classifier, "activation", "relu")),
        )
    raise TypeError(f"暂不支持转换为 PyTorch 头的分类器类型: {cls_name}")


class SklearnClassifierAdapter:
    """内存推理适配器：封装已加载的 scaler + sklearn 分类器，零磁盘 I/O。"""

    def __init__(self, scaler: Any, classifier: Any) -> None:
        self.scaler = scaler
        self.classifier = classifier
        self._torch_head: nn.Module | None = None
        try:
            self._torch_head = export_torch_head(scaler, classifier)
            self._torch_head.eval()
        except TypeError:
            self._torch_head = None

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float32)
        if self._torch_head is not None and type(self.classifier).__name__ == "SVC":
            with torch.no_grad():
                return self._torch_head(torch.as_tensor(feats)).numpy().astype(np.float64)
        z = self.scaler.transform(feats)
        if hasattr(self.classifier, "coef_") and not hasattr(self.classifier, "support_vectors_"):
            return np.asarray(self.classifier.decision_function(z), dtype=np.float64)
        p = np.clip(self.classifier.predict_proba(z), 1e-7, 1.0)
        return np.log(p).astype(np.float64)

    def predict_proba(self, features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        logits = self.predict_logits(features) / max(float(temperature), 1e-6)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(logits)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

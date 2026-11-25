"""覆冰类型四分类训练子包（CPU 特征缓存、SVM/LR/MLP 分类器训练、末端微调与编排）。"""

from __future__ import annotations

from grid_ice_vision.training.ice_type.feature_extractor import (
    configure_cpu_threads,
    extract_and_cache_features,
    load_feature_cache,
)
from grid_ice_vision.training.ice_type.finetune import FinetuneConfig, run_finetune
from grid_ice_vision.training.ice_type.train_svm import (
    ClassifierTrainConfig,
    train_cached_classifier,
)
from grid_ice_vision.training.ice_type.trainer import (
    load_ice_type_inferencer,
    run_training,
)

__all__ = [
    "ClassifierTrainConfig",
    "FinetuneConfig",
    "configure_cpu_threads",
    "extract_and_cache_features",
    "load_feature_cache",
    "load_ice_type_inferencer",
    "run_finetune",
    "run_training",
    "train_cached_classifier",
]

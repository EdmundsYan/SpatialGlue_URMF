#!/usr/bin/env python
"""
# Author: Yahui Long
# File Name: __init__.py
# Description:
"""

__author__ = "Yahui Long"
__email__ = "yahuilong1990@gmail.com"

# ✅ 导入修改后的模型组件
from .model import (
    Encoder_overall, 
    EncoderVAE,  # ✅ 新增
    reparameterize,  # ✅ 新增
)

# ✅ 导入预处理模块
from .preprocess import (
    adjacent_matrix_preprocessing, 
    fix_seed, 
    clr_normalize_each_cell, 
    lsi, 
    construct_neighbor_graph, 
    pca
)

# ✅ 导入工具函数
from .utils import (
    clustering, 
    plot_weight_value
)

# ✅ 导入 EURO 机制
from .train import apply_euro_gradient_adjustment  # ✅ 新增
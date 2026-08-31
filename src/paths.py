import os

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))

ALIGNED_TRAIN_PATH = os.path.join(SRC_ROOT, "cifake_aligned", "train")
WILDFAKE_ZIPS_DIR = os.path.join(SRC_ROOT, "wildfake_zips")
AIGC_OUT_DIR = os.path.join(SRC_ROOT, "wildfake_subset", "AIGC")
NON_AIGC_OUT_DIR = os.path.join(SRC_ROOT, "wildfake_subset", "Non-AIGC")
CHECKPOINT_PATH = os.path.join(SRC_ROOT, "checkpoint.pt")

TRAIN_FEATS_PATH = os.path.join(SRC_ROOT, "train_feats.pt")
VAL_FEATS_PATH = os.path.join(SRC_ROOT, "val_feats.pt")
TEST_FEATS_PATH = os.path.join(SRC_ROOT, "test_feats.pt")
TEST_ROBUST_FEATS_PATH = os.path.join(SRC_ROOT, "test_robust_feats.pt")
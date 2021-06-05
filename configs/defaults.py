from yacs.config import CfgNode as CN

from data.based import EncoderTypes, ScaleTypes
from model.based import MetricTypes, TrainingMode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# BASIC CONFIG
# -----------------------------------------------------------------------------
_C.BASIC = CN()
_C.BASIC.SEED = 2022
_C.BASIC.PCA = False
# -----------------------------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATASET_ADDRESS = '../data/dataset/bank.csv'
_C.DATASET.DATASET_BRIEF_DESCRIPTION = '../data/dataset/description.txt'
_C.DATASET.TARGET = 'y'
_C.DATASET.HAS_CATEGORICAL_TARGETS = True
# _C.DATASET.DROP_COLS = ('y')
# ---------------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------------- #
_C.SVM = CN()
_C.SVM.NAME = 'SVM'
_C.SVM.MODE = TrainingMode.CLASSIFICATION
_C.SVM.KERNEL = 'rbf'
_C.SVM.GAMMA = 'scale'
_C.SVM.VERBOSE = True

_C.RANDOM_FOREST = CN()
_C.RANDOM_FOREST.NAME = 'RANDOM_FOREST'
_C.RANDOM_FOREST.MODE = TrainingMode.CLASSIFICATION

_C.RANDOM_FOREST.n_estimators = 100
_C.RANDOM_FOREST.criterion = "gini"
_C.RANDOM_FOREST.max_depth = None
_C.RANDOM_FOREST.min_samples_split = 2
_C.RANDOM_FOREST.min_samples_leaf = 1
_C.RANDOM_FOREST.min_weight_fraction_leaf = 0.0
_C.RANDOM_FOREST.max_features = "auto"
_C.RANDOM_FOREST.max_leaf_nodes = None
_C.RANDOM_FOREST.min_impurity_decrease = 0.0
_C.RANDOM_FOREST.min_impurity_split = None
_C.RANDOM_FOREST.bootstrap = True
_C.RANDOM_FOREST.oob_score = False
_C.RANDOM_FOREST.n_jobs = None
_C.RANDOM_FOREST.random_state = None
_C.RANDOM_FOREST.verbose = 0
_C.RANDOM_FOREST.warm_start = False
_C.RANDOM_FOREST.class_weight = None
_C.RANDOM_FOREST.ccp_alpha = 0.0
_C.RANDOM_FOREST.max_samples = None

# ---------------------------------------------------------------------------- #
# metric
# ---------------------------------------------------------------------------- #
_C.METRIC = CN()

_C.METRIC = MetricTypes.F1_SCORE_MACRO
_C.CONFUSION_MATRIX = True

# -----------------------------------------------------------------------------
# CATEGORICAL FEATURES ENCODER CONFIG / _C.ENCODER.{COLUMN NAME} = TYPE OF ENCODER
# -----------------------------------------------------------------------------
_C.ENCODER = CN()
_C.ENCODER.JOB = EncoderTypes.LABEL
_C.ENCODER.MARITAL = EncoderTypes.LABEL
_C.ENCODER.EDUCATION = EncoderTypes.LABEL
_C.ENCODER.DEFAULT = EncoderTypes.LABEL
_C.ENCODER.HOUSING = EncoderTypes.LABEL
_C.ENCODER.LOAN = EncoderTypes.LABEL
_C.ENCODER.CONTACT = EncoderTypes.LABEL
_C.ENCODER.MONTH = EncoderTypes.LABEL
_C.ENCODER.POUTCOME = EncoderTypes.LABEL
# _C.ENCODER.Y = EncoderTypes.LABEL # if your target is categorical
# -----------------------------------------------------------------------------
# SCALER /
# -----------------------------------------------------------------------------
_C.SCALER = CN()
_C.SCALER = ScaleTypes.STANDARD

# -----------------------------------------------------------------------------
# DECOMPOSITION
# -----------------------------------------------------------------------------
_C.PCA = CN()
_C.PCA.N_COMPONENTS = 0.8

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# _C.OUTPUT_DIR = "../outputs"

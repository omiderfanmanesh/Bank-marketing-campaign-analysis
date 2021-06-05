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

_C.SVM.FINE_TUNE = CN()

# ---------------------------------------------------------------------------- #
_C.RANDOM_FOREST = CN()
_C.RANDOM_FOREST.NAME = 'RANDOM_FOREST'
_C.RANDOM_FOREST.MODE = TrainingMode.CLASSIFICATION

_C.RANDOM_FOREST.N_ESTIMATORS = 100  # default = 100
_C.RANDOM_FOREST.CRITERION = "gini"  # default = 'gini
_C.RANDOM_FOREST.MAX_DEPTH = None  # default = None
_C.RANDOM_FOREST.MIN_SAMPLES_SPLIT = 2  # default = 2
_C.RANDOM_FOREST.MIN_SAMPLES_LEAF = 1  # default = 1
_C.RANDOM_FOREST.MIN_WEIGHT_FRACTION_LEAF = 0.0  # default = 0.0
_C.RANDOM_FOREST.MAX_FEATURES = "auto"  # default =  "auto"
_C.RANDOM_FOREST.MAX_LEAF_NODES = None  # default = None
_C.RANDOM_FOREST.MIN_IMPURITY_DECREASE = 0.0  # default = 0.0
_C.RANDOM_FOREST.MIN_IMPURITY_SPLIT = None  # default = None
_C.RANDOM_FOREST.BOOTSTRAP = True  # default = True
_C.RANDOM_FOREST.OOB_SCORE = False  # default = False
_C.RANDOM_FOREST.N_JOBS = None  # default = None
_C.RANDOM_FOREST.RANDOM_STATE = None  # default = None
_C.RANDOM_FOREST.VERBOSE = 0  # default = 0
_C.RANDOM_FOREST.WARM_START = False  # default = False
_C.RANDOM_FOREST.CLASS_WEIGHT = None  # default = None
_C.RANDOM_FOREST.CCP_ALPHA = 0.0  # default = 0.0
_C.RANDOM_FOREST.MAX_SAMPLES = None  # default = None

_C.RANDOM_FOREST.FINE_TUNE = CN()
_C.RANDOM_FOREST.FINE_TUNE.N_ESTIMATORS = (5, 10, 15, 20)
_C.RANDOM_FOREST.FINE_TUNE.CRITERION = None

# ---------------------------------------------------------------------------- #
_C.LOGISTIC_REGRESSION = CN()
_C.LOGISTIC_REGRESSION.NAME = 'LOGISTIC REGRESSION'
_C.LOGISTIC_REGRESSION.MODE = TrainingMode.CLASSIFICATION  # 
_C.LOGISTIC_REGRESSION.PENALTY = 'l2'  # default = 'l2'
_C.LOGISTIC_REGRESSION.DUAL = False  # default = False
_C.LOGISTIC_REGRESSION.TOL = 1e-4  # default = 1e-4
_C.LOGISTIC_REGRESSION.C = 1.0  # default = 1.0 
_C.LOGISTIC_REGRESSION.FIT_INTERCEPT = True  # default = True
_C.LOGISTIC_REGRESSION.INTERCEPT_SCALING = 1  # default = 1
_C.LOGISTIC_REGRESSION.CLASS_WEIGHT = None  # default = None
_C.LOGISTIC_REGRESSION.RANDOM_STATE = None  # default = None
_C.LOGISTIC_REGRESSION.SOLVER = 'lbfgs'  # default = 'lbfgs'
_C.LOGISTIC_REGRESSION.MAX_ITER = 100  # default = 100
_C.LOGISTIC_REGRESSION.MULTI_CLASS = 'auto'  # default = 'auto'
_C.LOGISTIC_REGRESSION.VERBOSE = 0  # default = 0
_C.LOGISTIC_REGRESSION.WARM_START = False  # default = False
_C.LOGISTIC_REGRESSION.N_JOBS = None  # default = None
_C.LOGISTIC_REGRESSION.L1_RATIO = None  # default = None

_C.LOGISTIC_REGRESSION.FINE_TUNE = CN()

# ---------------------------------------------------------------------------- #
_C.DECISION_TREE = CN()
_C.DECISION_TREE.NAME = 'RANDOM_FOREST'
_C.DECISION_TREE.MODE = TrainingMode.CLASSIFICATION

_C.DECISION_TREE.CRITERION = "gini"  # criterion : {"gini", "entropy"}, default="gini"
_C.DECISION_TREE.SPLITTER = "best"  # splitter : {"best", "random"}, default="best"
_C.DECISION_TREE.MAX_DEPTH = None  # max_depth : int, default=None
_C.DECISION_TREE.MIN_SAMPLES_SPLIT = 2  # min_samples_split : int or float, default=2
_C.DECISION_TREE.MIN_SAMPLES_LEAF = 1  # min_samples_leaf : int or float, default=1
_C.DECISION_TREE.MIN_WEIGHT_FRACTION_LEAF = 0.0  # min_weight_fraction_leaf : float, default=0.0
_C.DECISION_TREE.MAX_FEATURES = None  # max_features : int, float or {"auto", "sqrt", "log2"}, default=None
_C.DECISION_TREE.RANDOM_STATE = None  # random_state : int, RandomState instance, default=None
_C.DECISION_TREE.MAX_LEAF_NODES = None  # max_leaf_nodes : int, default=None
_C.DECISION_TREE.MIN_IMPURITY_DECREASE = 0.0  # min_impurity_decrease : float, default=0.0
_C.DECISION_TREE.MIN_IMPURITY_SPLIT = None  # min_impurity_split : float, default=0
_C.DECISION_TREE.CLASS_WEIGHT = None  # class_weight : dict, list of dict or "balanced", default=None
_C.DECISION_TREE.PRESORT = 'deprecated'  # presort : deprecated, default='deprecated'
_C.DECISION_TREE.CCP_ALPHA = 0.0  # ccp_alpha : non-negative float, default=0.0

_C.DECISION_TREE.FINE_TUNE = CN()
_C.DECISION_TREE.FINE_TUNE.CRITERION = ("gini", "entropy")
_C.DECISION_TREE.FINE_TUNE.SPLITTER = None

# ---------------------------------------------------------------------------- #
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

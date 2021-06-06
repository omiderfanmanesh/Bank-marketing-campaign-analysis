from yacs.config import CfgNode as CN

from data.based import EncoderTypes, ScaleTypes
from model.based import MetricTypes, TrainingMode
from model.based import ModelSelection
from utils import RuntimeMode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# BASIC CONFIG
# -----------------------------------------------------------------------------
_C.BASIC = CN()
_C.BASIC.SEED = 2021
_C.BASIC.PCA = False
_C.BASIC.RAND_STATE = 2021
_C.BASIC.MODEL_SELECTION = ModelSelection.SVM
_C.BASIC.RUNTIME_MODE = RuntimeMode.CROSS_VAL

# -----------------------------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.K_FOLD = 5
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

_C.SVM.C = 1.0  # float, default=1.0
_C.SVM.KERNEL = 'rbf'  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
_C.SVM.DEGREE = 3  # int, default=3
_C.SVM.GAMMA = 'scale'  # {'scale', 'auto'} or float, default='scale'
_C.SVM.COEF0 = 0.0  # float, default=0.0
_C.SVM.SHRINKING = True  # bool, default=True
_C.SVM.PROBABILITY = False  # bool, default=False
_C.SVM.TOL = 1e-3  # float, default=1e-3
_C.SVM.CACHE_SIZE = 200  # float, default=200
_C.SVM.CLASS_WEIGHT = None  # dict or 'balanced', default=None
_C.SVM.VERBOSE = True  # bool, default=False
_C.SVM.MAX_ITER = -1  # int, default=-1
_C.SVM.DECISION_FUNCTION_SHAPE = 'ovr'  # {'ovo', 'ovr'}, default='ovr'
_C.SVM.BREAK_TIES = False  # bool, default=False
_C.SVM.RANDOM_STATE = _C.BASIC.RAND_STATE  # int or RandomState instance, default=None

_C.SVM.HYPER_PARAM_TUNING = CN()
_C.SVM.HYPER_PARAM_TUNING.KERNEL = ('linear', 'poly', 'rbf', 'sigmoid')
_C.SVM.HYPER_PARAM_TUNING.DEGREE = None
_C.SVM.HYPER_PARAM_TUNING.GAMMA = None
_C.SVM.HYPER_PARAM_TUNING.COEF0 = None
_C.SVM.HYPER_PARAM_TUNING.SHRINKING = None
_C.SVM.HYPER_PARAM_TUNING.PROBABILITY = None
_C.SVM.HYPER_PARAM_TUNING.TOL = None
_C.SVM.HYPER_PARAM_TUNING.CACHE_SIZE = None
_C.SVM.HYPER_PARAM_TUNING.CLASS_WEIGHT = None
_C.SVM.HYPER_PARAM_TUNING.MAX_ITER = None
_C.SVM.HYPER_PARAM_TUNING.DECISION_FUNCTION_SHAPE = None
_C.SVM.HYPER_PARAM_TUNING.BREAK_TIES = None

# ---------------------------------------------------------------------------- #
_C.RANDOM_FOREST = CN()
_C.RANDOM_FOREST.NAME = 'RANDOM_FOREST'
_C.RANDOM_FOREST.MODE = TrainingMode.CLASSIFICATION

_C.RANDOM_FOREST.N_ESTIMATORS = 100  # int, default=100
_C.RANDOM_FOREST.CRITERION = "gini"  # {"gini", "entropy"}, default="gini"
_C.RANDOM_FOREST.MAX_DEPTH = None  # int, default=None
_C.RANDOM_FOREST.MIN_SAMPLES_SPLIT = 2  # int or float, default=2
_C.RANDOM_FOREST.MIN_SAMPLES_LEAF = 1  # int or float, default=1
_C.RANDOM_FOREST.MIN_WEIGHT_FRACTION_LEAF = 0.0  # float, default=0.0
_C.RANDOM_FOREST.MAX_FEATURES = "auto"  # {"auto", "sqrt", "log2"}, int or float, default="auto"
_C.RANDOM_FOREST.MAX_LEAF_NODES = None  # int, default=None
_C.RANDOM_FOREST.MIN_IMPURITY_DECREASE = 0.0  # float, default=0.0
_C.RANDOM_FOREST.MIN_IMPURITY_SPLIT = None  # float, default=None
_C.RANDOM_FOREST.BOOTSTRAP = True  # bool, default=True
_C.RANDOM_FOREST.OOB_SCORE = False  # bool, default=False
_C.RANDOM_FOREST.N_JOBS = None  # int, default=None
_C.RANDOM_FOREST.RANDOM_STATE = _C.BASIC.RAND_STATE  # int or RandomState, default=None
_C.RANDOM_FOREST.VERBOSE = 0  # int, default=0
_C.RANDOM_FOREST.WARM_START = False  # bool, default=False
_C.RANDOM_FOREST.CLASS_WEIGHT = None  # "balanced", "balanced_subsample"}, dict or list of dicts, default=None
_C.RANDOM_FOREST.CCP_ALPHA = 0.0  # non-negative float, default=0.0
_C.RANDOM_FOREST.MAX_SAMPLES = None  # int or float, default=None

_C.RANDOM_FOREST.HYPER_PARAM_TUNING = CN()
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.N_ESTIMATORS = (5, 10, 15, 20)
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.CRITERION = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MAX_DEPTH = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_SAMPLES_SPLIT = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_SAMPLES_LEAF = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_WEIGHT_FRACTION_LEAF = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MAX_FEATURES = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MAX_LEAF_NODES = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_IMPURITY_DECREASE = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_IMPURITY_SPLIT = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.BOOTSTRAP = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.OOB_SCORE = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.WARM_START = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.CLASS_WEIGHT = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.CCP_ALPHA = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MAX_SAMPLES = None

# ---------------------------------------------------------------------------- #
_C.LOGISTIC_REGRESSION = CN()
_C.LOGISTIC_REGRESSION.NAME = 'LOGISTIC REGRESSION'
_C.LOGISTIC_REGRESSION.MODE = TrainingMode.CLASSIFICATION  # 
_C.LOGISTIC_REGRESSION.PENALTY = 'l2'  # {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
_C.LOGISTIC_REGRESSION.DUAL = False  # bool, default=False
_C.LOGISTIC_REGRESSION.TOL = 1e-4  # float, default=1e-4
_C.LOGISTIC_REGRESSION.C = 1.0  # float, default=1.0
_C.LOGISTIC_REGRESSION.FIT_INTERCEPT = True  # bool, default=True
_C.LOGISTIC_REGRESSION.INTERCEPT_SCALING = 1  # float, default=1
_C.LOGISTIC_REGRESSION.CLASS_WEIGHT = None  # dict or 'balanced', default=None
_C.LOGISTIC_REGRESSION.RANDOM_STATE = _C.BASIC.RAND_STATE  # int, RandomState instance, default=None
_C.LOGISTIC_REGRESSION.SOLVER = 'lbfgs'  # {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'
_C.LOGISTIC_REGRESSION.MAX_ITER = 100  # int, default=100
_C.LOGISTIC_REGRESSION.MULTI_CLASS = 'auto'  # {'auto', 'ovr', 'multinomial'}, default='auto'
_C.LOGISTIC_REGRESSION.VERBOSE = 0  # int, default=0
_C.LOGISTIC_REGRESSION.WARM_START = False  # bool, default=False
_C.LOGISTIC_REGRESSION.N_JOBS = None  # int, default=None
_C.LOGISTIC_REGRESSION.L1_RATIO = None  # float, default=None

_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING = CN()
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.PENALTY = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.DUAL = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.TOL = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.C = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.FIT_INTERCEPT = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.INTERCEPT_SCALING = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.CLASS_WEIGHT = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.SOLVER = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.MAX_ITER = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.MULTI_CLASS = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.WARM_START = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.L1_RATIO = None

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
_C.DECISION_TREE.RANDOM_STATE = _C.BASIC.RAND_STATE  # random_state : int, RandomState instance, default=None
_C.DECISION_TREE.MAX_LEAF_NODES = None  # max_leaf_nodes : int, default=None
_C.DECISION_TREE.MIN_IMPURITY_DECREASE = 0.0  # min_impurity_decrease : float, default=0.0
_C.DECISION_TREE.MIN_IMPURITY_SPLIT = None  # min_impurity_split : float, default=0
_C.DECISION_TREE.CLASS_WEIGHT = None  # class_weight : dict, list of dict or "balanced", default=None
_C.DECISION_TREE.PRESORT = 'deprecated'  # presort : deprecated, default='deprecated'
_C.DECISION_TREE.CCP_ALPHA = 0.0  # ccp_alpha : non-negative float, default=0.0

_C.DECISION_TREE.FINE_TUNE = CN()
_C.DECISION_TREE.FINE_TUNE.CRITERION = ("gini", "entropy")
_C.DECISION_TREE.FINE_TUNE.SPLITTER = None
_C.DECISION_TREE.FINE_TUNE.MAX_DEPTH = None
_C.DECISION_TREE.FINE_TUNE.MIN_SAMPLES_SPLIT = None
_C.DECISION_TREE.FINE_TUNE.MIN_SAMPLES_LEAF = None
_C.DECISION_TREE.FINE_TUNE.MIN_WEIGHT_FRACTION_LEAF = None
_C.DECISION_TREE.FINE_TUNE.MAX_FEATURES = None
_C.DECISION_TREE.FINE_TUNE.MAX_LEAF_NODES = None
_C.DECISION_TREE.FINE_TUNE.MIN_IMPURITY_DECREASE = None
_C.DECISION_TREE.FINE_TUNE.MIN_IMPURITY_SPLIT = None
_C.DECISION_TREE.FINE_TUNE.CLASS_WEIGHT = None
_C.DECISION_TREE.FINE_TUNE.PRESORT = None
_C.DECISION_TREE.FINE_TUNE.CCP_ALPHA = None
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
_C.ENCODER.Y = EncoderTypes.LABEL  # if your target is categorical
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

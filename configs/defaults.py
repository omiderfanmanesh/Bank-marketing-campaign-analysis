from yacs.config import CfgNode as CN

from data.based import EncoderTypes, ScaleTypes, Sampling
from model.based import MetricTypes, TaskMode
from model.based import Model
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
_C.BASIC.MODEL = Model.SVM
_C.BASIC.RUNTIME_MODE = RuntimeMode.TRAIN
_C.BASIC.TASK_MODE = TaskMode.CLASSIFICATION
_C.BASIC.SAMPLING_STRATEGY = (Sampling.SMOTE, Sampling.RANDOM_UNDER_SAMPLING)  # None means don't use resampling
# -----------------------------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.K_FOLD = 5

# -----------------------------------------------------------------------------
# SAMPLING
# -----------------------------------------------------------------------------
_C.RANDOM_UNDER_SAMPLER = CN()
_C.RANDOM_UNDER_SAMPLER.SAMPLING_STRATEGY = 'auto'  # float, str, dict, callable, default=’auto’
_C.RANDOM_UNDER_SAMPLER.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.RANDOM_UNDER_SAMPLER.REPLACEMENT = False  # bool, default=False

_C.RANDOM_OVER_SAMPLER = CN()
_C.RANDOM_OVER_SAMPLER.SAMPLING_STRATEGY = 'minority'  # float, str, dict or callable, default=’auto’
_C.RANDOM_OVER_SAMPLER.RANDOM_STATE = 2021  # int, RandomState instance, default=None
# _C.RANDOM_OVER_SAMPLER.SHRINKAGE = 0  # float or dict, default=None

_C.SMOTE = CN()
_C.SMOTE.SAMPLING_STRATEGY = 'auto'  # float, str, dict or callable, default=’auto’ {'minority'}
_C.SMOTE.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.SMOTE.K_NEIGHBORS = 5  # int or object, default=5
_C.SMOTE.N_JOBS = -1  # int, default=None

_C.SMOTENC = CN()
_C.SMOTENC.CATEGORICAL_FEATURES = ('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                                   'month')  # ndarray of shape (n_cat_features,) or (n_features,)
_C.SMOTENC.SAMPLING_STRATEGY = 'minority'  # float, str, dict or callable, default=’auto’
_C.SMOTENC.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.SMOTENC.K_NEIGHBORS = 5  # int or object, default=5
_C.SMOTENC.N_JOBS = -1  # int, default=None

_C.SVMSMOTE = CN()
_C.SVMSMOTE.SAMPLING_STRATEGY = 'auto'  # float, str, dict or callable, default=’auto’ {'minority'}
_C.SVMSMOTE.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.SVMSMOTE.K_NEIGHBORS = 3  # int or object, default=5
_C.SVMSMOTE.N_JOBS = -1  # int, default=None
_C.SVMSMOTE.M_NEIGHBORS = 10  # int or object, default=10
# _C.SVMSMOTE.SVM_ESTIMATOR = 5  # estimator object, default=SVC()
_C.SVMSMOTE.OUT_STEP = 0.5  # float, default=0.5

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
# metric
# ---------------------------------------------------------------------------- #
_C.EVALUATION = CN()

_C.EVALUATION.METRIC = MetricTypes.F1_SCORE_MICRO
_C.EVALUATION.CONFUSION_MATRIX = True
"""
'accuracy', 'balanced_accuracy',  'top_k_accuracy',
 'average_precision',  'neg_brier_score', 'f1',
 'f1_micro', 'f1_macro',  'f1_weighted',
 'f1_samples',  'neg_log_loss', 'precision',
  'recall',  'jaccard', 'roc_auc',
 'roc_auc_ovr', 'roc_auc_ovo',  'roc_auc_ovr_weighted',
 'roc_auc_ovo_weighted'
"""
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


# ---------------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------------- #
_C.SVM = CN()
_C.SVM.NAME = 'SVM'

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

_C.SVR = CN()
_C.SVR.NAME = 'SVM'

_C.SVR.KERNEL = 'rbf'  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
_C.SVR.DEGREE = 3  # int, default=3
_C.SVR.GAMMA = 'scale'  # {'scale', 'auto'} or float, default='scale'
_C.SVR.COEF0 = 0.0  # float, default=0.0
_C.SVR.TOL = 1e-3  # float, default=1e-3
_C.SVR.C = 1.0  # float, default=1.0
_C.SVR.EPSILON = 0.1  # float, default=0.1
_C.SVR.SHRINKING = True  # bool, default=True
_C.SVR.CACHE_SIZE = 200  # float, default=200
_C.SVR.VERBOSE = True  # bool, default=False
_C.SVR.MAX_ITER = -1  # int, default=-1

_C.SVM.HYPER_PARAM_TUNING = CN()
_C.SVM.HYPER_PARAM_TUNING.KERNEL = ('linear', 'poly', 'rbf', 'sigmoid')
_C.SVM.HYPER_PARAM_TUNING.C = (0.1, 1, 10, 100, 1000)
_C.SVM.HYPER_PARAM_TUNING.DEGREE = (1, 2, 3, 4, 5)
_C.SVM.HYPER_PARAM_TUNING.GAMMA = ('scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001)
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

_C.LOGISTIC_REGRESSION.PENALTY = 'l2'  # {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
_C.LOGISTIC_REGRESSION.DUAL = False  # bool, default=False
_C.LOGISTIC_REGRESSION.TOL = 1e-4  # float, default=1e-4
_C.LOGISTIC_REGRESSION.C = 1.0  # float, default=1.0
_C.LOGISTIC_REGRESSION.FIT_INTERCEPT = True  # bool, default=True
_C.LOGISTIC_REGRESSION.INTERCEPT_SCALING = 1  # float, default=1
_C.LOGISTIC_REGRESSION.CLASS_WEIGHT = None  # dict or 'balanced', default=None
_C.LOGISTIC_REGRESSION.RANDOM_STATE = _C.BASIC.RAND_STATE  # int, RandomState instance, default=None
_C.LOGISTIC_REGRESSION.SOLVER = 'lbfgs'  # {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'
_C.LOGISTIC_REGRESSION.MAX_ITER = 10000  # int, default=100
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
_C.DECISION_TREE.NAME = 'DECISION TREE'

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

_C.DECISION_TREE.HYPER_PARAM_TUNING = CN()
_C.DECISION_TREE.HYPER_PARAM_TUNING.CRITERION = ("gini", "entropy")
_C.DECISION_TREE.HYPER_PARAM_TUNING.SPLITTER = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MAX_DEPTH = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_SAMPLES_SPLIT = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_SAMPLES_LEAF = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_WEIGHT_FRACTION_LEAF = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MAX_FEATURES = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MAX_LEAF_NODES = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_IMPURITY_DECREASE = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_IMPURITY_SPLIT = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.CLASS_WEIGHT = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.PRESORT = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.CCP_ALPHA = None
# ---------------------------------------------------------------------------- #

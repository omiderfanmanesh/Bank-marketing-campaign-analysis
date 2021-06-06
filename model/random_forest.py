from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from model.based import BasedModel, TrainingMode


class RandomForest(BasedModel):
    def __init__(self, cfg):
        super(RandomForest, self).__init__(cfg=cfg)
        self._params = {

            'n_estimators': cfg.RANDOM_FOREST.N_ESTIMATORS,
            'criterion': cfg.RANDOM_FOREST.CRITERION,
            'max_depth': cfg.RANDOM_FOREST.MAX_DEPTH,
            'min_samples_split': cfg.RANDOM_FOREST.MIN_SAMPLES_SPLIT,
            'min_samples_leaf': cfg.RANDOM_FOREST.MIN_SAMPLES_LEAF,
            'min_weight_fraction_leaf': cfg.RANDOM_FOREST.MIN_WEIGHT_FRACTION_LEAF,
            'max_features': cfg.RANDOM_FOREST.MAX_FEATURES,
            'max_leaf_nodes': cfg.RANDOM_FOREST.MAX_LEAF_NODES,
            'min_impurity_decrease': cfg.RANDOM_FOREST.MIN_IMPURITY_DECREASE,

            'min_impurity_split': cfg.RANDOM_FOREST.MIN_IMPURITY_SPLIT,
            'bootstrap': cfg.RANDOM_FOREST.BOOTSTRAP,
            'oob_score': cfg.RANDOM_FOREST.OOB_SCORE,
            'n_jobs': cfg.RANDOM_FOREST.N_JOBS,
            'random_state': cfg.RANDOM_FOREST.RANDOM_STATE,
            'verbose': cfg.RANDOM_FOREST.VERBOSE,

            'warm_start': cfg.RANDOM_FOREST.WARM_START,
            'class_weight': cfg.RANDOM_FOREST.CLASS_WEIGHT,
            'ccp_alpha': cfg.RANDOM_FOREST.CCP_ALPHA,
            'max_samples': cfg.RANDOM_FOREST.MAX_SAMPLES,

        }

        self._training_mode = cfg.RANDOM_FOREST.MODE
        if self._training_mode == TrainingMode.CLASSIFICATION:
            self.model = RandomForestClassifier(**self._params)
        elif self._training_mode == TrainingMode.REGRESSION:
            self.model = RandomForestRegressor(**self._params)

        for _k in cfg.RANDOM_FOREST.HYPER_PARAM_TUNING:
            _param = cfg.RANDOM_FOREST.HYPER_PARAM_TUNING[_k]

            if _param is not None:
                _param = [*_param]
                self.fine_tune_params[_k.lower()] = [*_param]

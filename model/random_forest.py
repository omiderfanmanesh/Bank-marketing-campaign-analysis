from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from model.based import BasedModel, TrainingMode


class RandomForest(BasedModel):
    def __init__(self, cfg):
        super(RandomForest, self).__init__(cfg=cfg)
        self._params = {

            'n_estimators': cfg.RANDOM_FOREST.n_estimators,
            'criterion': cfg.RANDOM_FOREST.criterion,
            'max_depth': cfg.RANDOM_FOREST.max_depth,
            'min_samples_split': cfg.RANDOM_FOREST.min_samples_split,
            'min_samples_leaf': cfg.RANDOM_FOREST.min_samples_leaf,
            'min_weight_fraction_leaf': cfg.RANDOM_FOREST.min_weight_fraction_leaf,
            'max_features': cfg.RANDOM_FOREST.max_features,
            'max_leaf_nodes': cfg.RANDOM_FOREST.max_leaf_nodes,
            'min_impurity_decrease': cfg.RANDOM_FOREST.min_impurity_decrease,

            'min_impurity_split': cfg.RANDOM_FOREST.min_impurity_split,
            'bootstrap': cfg.RANDOM_FOREST.bootstrap,
            'oob_score': cfg.RANDOM_FOREST.oob_score,
            'n_jobs': cfg.RANDOM_FOREST.n_jobs,
            'random_state': cfg.RANDOM_FOREST.random_state,
            'verbose': cfg.RANDOM_FOREST.verbose,

            'warm_start': cfg.RANDOM_FOREST.warm_start,
            'class_weight': cfg.RANDOM_FOREST.class_weight,
            'ccp_alpha': cfg.RANDOM_FOREST.ccp_alpha,
            'max_samples': cfg.RANDOM_FOREST.max_samples,

        }

        self._training_mode = cfg.RANDOM_FOREST.MODE
        if self._training_mode == TrainingMode.CLASSIFICATION:
            self.model = RandomForestClassifier(**self._params)
        elif self._training_mode == TrainingMode.REGRESSION:
            self.model = RandomForestRegressor(**self._params)

    def get_model(self):
        return self.model

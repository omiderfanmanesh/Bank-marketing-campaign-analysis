def do_train(cfg, model, dataset):
    X_train, X_test, y_train, y_test = dataset.split_to(has_validation=False)
    model.train(X_train=X_train, y_train=y_train)
    model.evaluate(X_test=X_test, y_test=y_test)

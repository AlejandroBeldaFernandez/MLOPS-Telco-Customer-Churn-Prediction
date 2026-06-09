from src.evaluation import evaluation


def test_dict(pipeline):
    pipeline_final, X_train, X_test, y_train, y_test = pipeline
    metrics = evaluation(X_train, X_test, y_train, y_test, pipeline_final)
    assert isinstance(metrics, dict)


def test_metrics_keys(pipeline):
    pipeline_final, X_train, X_test, y_train, y_test = pipeline
    metrics = evaluation(X_train, X_test, y_train, y_test, pipeline_final)
    expected_keys = {
        "balanced_accuracy_train",
        "roc_train",
        "precision_train",
        "recall_train",
        "balanced_accuracy_test",
        "roc_test",
        "precision_test",
        "recall_test",
    }
    assert set(metrics.keys()) == expected_keys


def test_metrics_values(pipeline):
    pipeline_final, X_train, X_test, y_train, y_test = pipeline
    metrics = evaluation(X_train, X_test, y_train, y_test, pipeline_final)
    for value in metrics.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

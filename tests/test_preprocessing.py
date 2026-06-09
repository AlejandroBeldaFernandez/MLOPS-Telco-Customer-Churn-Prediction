from src.preprocessing import preprocessing


def test_customer_id_dropped(sample_df):
    X_train, X_test, y_train, y_test, preprocessor = preprocessing(sample_df)
    assert "customerID" not in X_train.columns


def test_avg_monthly_charges_engineered(sample_df):
    X_train, X_test, y_train, y_test, preprocessor = preprocessing(sample_df)
    assert "avg_monthly_charges" in X_train.columns


def test_y_binary(sample_df):
    X_train, X_test, y_train, y_test, preprocessor = preprocessing(sample_df)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_churn_dropped(sample_df):
    X_train, X_test, y_train, y_test, preprocessor = preprocessing(sample_df)
    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns

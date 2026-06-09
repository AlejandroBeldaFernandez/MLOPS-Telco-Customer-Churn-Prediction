from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.monitor import monitoring, send_alert_email


# Simple DataFrames with numeric columns — no need for the full Telco schema
@pytest.fixture
def drift_data():
    rng = np.random.default_rng(42)
    reference_df = pd.DataFrame({"col1": rng.normal(0, 1, 50), "col2": rng.normal(0, 1, 50)})
    current_df = pd.DataFrame({"col1": rng.normal(0, 1, 50), "col2": rng.normal(0, 1, 50)})
    return reference_df, current_df


def test_monitoring_returns_bool(drift_data):
    reference_df, current_df = drift_data

    with patch("src.monitor.psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value = MagicMock()
        result = monitoring(reference_df, current_df)

    assert isinstance(result, bool)


def test_monitoring_inserts_to_db(drift_data):
    """Verifica que se inserta exactamente 1 fila en la BD por cada ejecución."""
    reference_df, current_df = drift_data
    mock_cursor = MagicMock()

    with patch("src.monitor.psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value = mock_cursor
        monitoring(reference_df, current_df)

    mock_cursor.execute.assert_called_once()
    mock_connect.return_value.commit.assert_called_once()


def test_send_alert_email_calls_smtp():
    """Verifica que se llama a starttls, login y send_message sin enviar un email real."""
    mock_server = MagicMock()

    with patch("src.monitor.smtplib.SMTP") as mock_smtp_class:
        # smtplib.SMTP se usa como context manager: with smtplib.SMTP(...) as server
        mock_smtp_class.return_value.__enter__.return_value = mock_server

        with patch.dict("os.environ", {"GMAIL_USER": "test@test.com", "GMAIL_PASSWORD": "fakepass"}):
            send_alert_email({"ROC_AUC_test": 0.85})

    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once()
    mock_server.send_message.assert_called_once()

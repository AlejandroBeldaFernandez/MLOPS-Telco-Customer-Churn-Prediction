import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

load_dotenv()


def monitoring(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
    """
    Usa Evidently AI para comparar el dataset de referencia (entrenamiento)
    con el dataset actual (producción). Retorna True si se detecta Drift.
    """
    logging.info("-----------------------------------------------------------------------")

    data_drift_report = Report(metrics=[DataDriftPreset(drift_share=0.4)])
    data_drift_report.run(reference_data=reference_df, current_data=current_df)

    report_dict = data_drift_report.as_dict()

    drift_share = report_dict["metrics"][0]["result"]["share_of_drifted_columns"]
    number_drifted_columns = report_dict["metrics"][0]["result"]["number_of_drifted_columns"]
    number_columns = report_dict["metrics"][0]["result"]["number_of_columns"]
    dataset_drifted = report_dict["metrics"][0]["result"]["dataset_drift"]
    logging.info(f"Porcentaje de columnas con Drift detectado: {drift_share * 100:.2f}%")
    logging.info(f"Número de columnas con Drift: {number_drifted_columns}")
    logging.info(f"Número total de columnas: {number_columns}")
    logging.info(f"Dataset con Drift: {dataset_drifted}")

    conn = psycopg2.connect(host="localhost", port=5433, database="mydb", user="admin", password="admin")
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO drift_report 
        (timestamp, drift_share, n_drifted_columns, n_columns, drift_detected)
        VALUES (%s, %s, %s, %s, %s)""",
        (
            datetime.now(),
            drift_share,
            number_drifted_columns,
            number_columns,
            dataset_drifted,
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return dataset_drifted


def send_alert_email(metrics: dict):
    msg = MIMEText(f"New model trained. Metrics of the model:\n\n{metrics}")
    msg["Subject"] = "MLOps Alert: Model Training Completed"
    msg["From"] = os.getenv("GMAIL_USER")
    msg["To"] = os.getenv("GMAIL_USER")

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(os.getenv("GMAIL_USER"), os.getenv("GMAIL_PASSWORD"))
        server.send_message(msg)

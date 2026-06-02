CREATE TABLE drift_report (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    drift_share FLOAT NOT NULL,
    n_drifted_columns INTEGER NOT NULL,
    n_columns INTEGER NOT NULL,
    drift_detected BOOLEAN NOT NULL
    
);
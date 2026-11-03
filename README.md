# Log Anomaly Detection (HDFS_v1)

System logs from large clusters are noisy and high-volume, so rare failures are easy to miss by hand. This project detects anomalous sequences in the HDFS_v1 log dataset: we turn raw HDFS logs into structured event sequences, train a model to learn normal behavior, and flag unusual patterns that may signal faults or attacks. The goal is a reproducible pipeline (data prep → train → evaluate) with a small FastAPI service for scoring new log batches.

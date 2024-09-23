# app/utils/logger.py
from tinydb import TinyDB, Query
from datetime import datetime

db = TinyDB("model_usage_logs.json")
metrics_db = TinyDB("model_metrics.json")


def log_model_usage(model_name, model_version):
    log_entry = {
        "model_name": model_name,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat(),
    }
    db.insert(log_entry)


def update_model_metrics(model_name, metrics):
    Model = Query()
    metrics_db.upsert(
        {"model_name": model_name, "metrics": metrics}, Model.model_name == model_name
    )


def get_model_metrics(model_name):
    Model = Query()
    result = metrics_db.search(Model.model_name == model_name)
    return result[0]["metrics"] if result else None


def get_model_usage_logs():
    return db.all()

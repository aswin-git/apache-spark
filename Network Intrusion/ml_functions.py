# ── ADD TO etl_functions.py ───────────────────────────────────────

from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import when, col
from pyspark.sql.types import NumericType


def prepare_features(df):
    """
    Assemble numeric columns into ML feature vector.
    Adds classWeight column to handle class imbalance.
    """
    exclude_cols = [
        "Label", "label_index", "is_attack",
        "source_file", "duration_bucket",
        "features_raw", "features"
    ]
    feature_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, NumericType)
        and f.name not in exclude_cols
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )
    df = assembler.transform(df)

    # Class weights for imbalance (83% benign vs 17% attack)
    total        = df.count()
    benign_count = df.filter(col("is_attack") == 0).count()
    attack_count = df.filter(col("is_attack") == 1).count()

    w_benign = total / (2.0 * benign_count)
    w_attack = total / (2.0 * attack_count)

    df = df.withColumn("classWeight",
        when(col("is_attack") == 0, w_benign).otherwise(w_attack))

    log.info(f"Features assembled: {len(feature_cols)} columns")
    log.info(f"Class weights — BENIGN: {w_benign:.4f}  ATTACK: {w_attack:.4f}")
    return df, feature_cols


def split_data(df, train_ratio=0.8, seed=42):
    """Split into train and test sets."""
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    log.info(f"Train: {train_df.count():,}  Test: {test_df.count():,}")
    return train_df, test_df


def train_model(train_df, model_dir):
    """
    Train Random Forest classifier and save to disk.
    Uses classWeight to handle class imbalance.
    """
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="is_attack",
        weightCol="classWeight",
        numTrees=100,
        maxDepth=15,
        maxBins=32,
        seed=42
    )
    log.info("Training Random Forest model...")
    model = rf.fit(train_df)
    model.save(model_dir)
    log.info(f"Model saved to {model_dir}")
    return model


def evaluate_model(predictions, model_name="Model"):
    """Evaluate predictions and log all metrics."""
    def get_metric(metric_name, evaluator_class, **kwargs):
        return evaluator_class(
            labelCol="is_attack",
            predictionCol="prediction",
            **kwargs
        ).evaluate(predictions)

    accuracy  = MulticlassClassificationEvaluator(
        labelCol="is_attack", predictionCol="prediction",
        metricName="accuracy").evaluate(predictions)
    precision = MulticlassClassificationEvaluator(
        labelCol="is_attack", predictionCol="prediction",
        metricName="weightedPrecision").evaluate(predictions)
    recall    = MulticlassClassificationEvaluator(
        labelCol="is_attack", predictionCol="prediction",
        metricName="weightedRecall").evaluate(predictions)
    f1        = MulticlassClassificationEvaluator(
        labelCol="is_attack", predictionCol="prediction",
        metricName="f1").evaluate(predictions)
    auc       = BinaryClassificationEvaluator(
        labelCol="is_attack", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC").evaluate(predictions)

    log.info(f"=== {model_name} Results ===")
    log.info(f"Accuracy  : {accuracy:.4f}")
    log.info(f"Precision : {precision:.4f}")
    log.info(f"Recall    : {recall:.4f}")
    log.info(f"F1 Score  : {f1:.4f}")
    log.info(f"AUC-ROC   : {auc:.4f}")

    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "auc": auc}


def predict_and_save(model, df, output_dir):
    """
    Run predictions on new data and save results to Parquet.
    Output includes original features + prediction + probability.
    """
    predictions = model.transform(df)

    # Extract attack probability from probability vector
    from pyspark.ml.functions import vector_to_array
    predictions = predictions.withColumn(
        "attack_probability",
        vector_to_array(col("probability")).getItem(1)
    )

    # Save only key columns
    result_cols = [
        "Label", "is_attack", "prediction",
        "attack_probability", "source_file"
    ]
    predictions.select(result_cols) \
        .repartition(4) \
        .write.mode("overwrite") \
        .parquet(output_dir + "predictions/")

    log.info(f"Predictions saved to {output_dir}predictions/")
    return predictions


def load_model(model_dir):
    """Load a previously saved Random Forest model."""
    model = RandomForestClassificationModel.load(model_dir)
    log.info(f"Model loaded from {model_dir}")
    return model


def feature_importance(model, feature_cols, top_n=15):
    """Log top N most important features."""
    import pandas as pd
    fi = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": list(model.featureImportances)
    }).sort_values("Importance", ascending=False)
    log.info(f"\nTop {top_n} Features:\n{fi.head(top_n).to_string(index=False)}")
    return fi
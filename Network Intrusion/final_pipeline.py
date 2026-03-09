# pipeline

from pyspark.sql import SparkSession
from etl_functions import (
    read_csvs,
    clean_column_names, handle_missing_values, replace_infinity,
    remove_duplicates, cap_outliers, encode_labels,
    engineer_features, scale_features,
    write_parquet, write_mysql, write_label_summary,
    cleanup, validate_output
)
from ml_functions import (
    # Extract
    read_csvs,
    # Transform
    clean_column_names, handle_missing_values, replace_infinity,
    remove_duplicates, cap_outliers, encode_labels,
    engineer_features, scale_features,
    # Load
    write_parquet, write_mysql, write_label_summary,
    # ML  ← new imports
    prepare_features, split_data, train_model,
    evaluate_model, predict_and_save, load_model,
    feature_importance,
    # Utils
    validate_output, cleanup
)
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CICIDS_ETL")


class CICIDS_ETL_Pipeline:

    def __init__(self, input_dir, output_dir):
        self.input_dir  = input_dir
        self.output_dir = output_dir
        self.model_dir  = model_dir          # ← new
        self.spark      = self._create_session()

    def _create_session(self):
        return SparkSession.builder \
            .appName("CICIDS2017_ETL") \
            .master("local[3]") \
            .config("spark.driver.memory", "10g") \
            .config("spark.driver.maxResultSize", "4g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "2g") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.default.parallelism", "6") \
            .config("spark.serializer",
                    "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.local.dir", "/tmp/spark-temp") \
            .config("spark.worker.cleanup.enabled", "true") \
            .config("spark.worker.cleanup.interval", "300") \
            .getOrCreate()

    def extract(self):
        return read_csvs(self.spark, self.input_dir)

    def transform(self, df):
        df = clean_column_names(df)
        df = handle_missing_values(df)
        df = replace_infinity(df)
        df = remove_duplicates(df)
        df = cap_outliers(df, column="Flow Duration")
        df = encode_labels(df)
        df = engineer_features(df)
        df = scale_features(df)
        return df

    def load(self, df):
        write_parquet(df, self.output_dir)
        # write_mysql(df, self.output_dir)
        write_label_summary(df, self.output_dir)

    def train(self, df):
        """Step ML.1 — Prepare, split, train, evaluate."""
        # Prepare features and class weights
        df_ml, feature_cols = prepare_features(df)

        # Split data
        train_df, test_df = split_data(df_ml, train_ratio=0.8)

        # Train model
        model = train_model(train_df, self.model_dir)

        # Evaluate on test set
        predictions = model.transform(test_df)
        metrics = evaluate_model(predictions, "Random Forest")

        # Feature importance
        feature_importance(model, feature_cols)

        # Save predictions
        predict_and_save(model, test_df, self.output_dir)

        return model, metrics

    def predict(self, df):
        """Step ML.2 — Load saved model and predict on new data."""
        model = load_model(self.model_dir)
        df_ml, _ = prepare_features(df)
        predictions = predict_and_save(model, df_ml, self.output_dir)
        return predictions

    def run(self):
        log.info("=== CICIDS2017 ETL + ML Pipeline Starting ===")

        # ── ETL ───────────────────────────────────────────────────
        log.info("--- Phase 1: Extract ---")
        df = self.extract()

        log.info("--- Phase 2: Transform ---")
        df = self.transform(df)

        log.info("--- Phase 3: Load ---")
        self.load(df)

        # ── ML ────────────────────────────────────────────────────
        log.info("--- Phase 4: Train ML Model ---")
        model, metrics = self.train(df)

        # ── Validate ──────────────────────────────────────────────
        log.info("--- Phase 5: Validate ---")
        validate_output(self.spark, self.output_dir)

        # ── Cleanup ───────────────────────────────────────────────
        cleanup(self.spark)

        log.info("=== Pipeline Complete ===")
        log.info(f"Final F1 Score : {metrics['f1']:.4f}")
        log.info(f"Final AUC-ROC  : {metrics['auc']:.4f}")


if __name__ == "__main__":
    pipeline = CICIDS_ETL_Pipeline(
        input_dir  = "./dataset/",
        output_dir = "./output/cleaned/",
        model_dir  = "./output/models/random_forest/"
    )
    pipeline.run()
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
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CICIDS_ETL")


class CICIDS_ETL_Pipeline:

    def __init__(self, input_dir, output_dir):
        self.input_dir  = input_dir
        self.output_dir = output_dir
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

    def run(self):
        log.info("=== CICIDS2017 ETL Pipeline Starting ===")
        df = self.extract()
        df = self.transform(df)
        self.load(df)
        validate_output(self.spark, self.output_dir)
        cleanup(self.spark)
        log.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    pipeline = CICIDS_ETL_Pipeline("./dataset/", "./output/cleaned/")
    pipeline.run()
# ─────────────────────────────────────────────
# etl_functions.py  ← all standalone functions
# ─────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, trim, input_file_name, regexp_extract, percentile_approx
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql.types import NumericType
import logging, shutil, os

log = logging.getLogger("CICIDS_ETL")


# ── EXTRACT ───────────────────────────────────────────────────────

def read_csvs(spark, input_dir):
    """Read all CSV files and strip column name spaces."""
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("mode", "DROPMALFORMED") \
        .csv(input_dir + "*.csv")
    df = df.toDF(*[c.strip() for c in df.columns])
    df = df.withColumn("source_file",
        regexp_extract(input_file_name(), r"(\w+-WorkingHours)", 1))
    log.info(f"Extracted {df.count():,} rows, {len(df.columns)} columns")
    return df


# ── TRANSFORM ─────────────────────────────────────────────────────

def clean_column_names(df):
    df = df.toDF(*[c.strip() for c in df.columns])
    # Fix: use the actual name from your CSV
    df = df.withColumnRenamed("Fwd Header Length34", "Fwd Header Length2")
    log.info("Column names cleaned.")
    return df


def handle_missing_values(df):
    """Drop rows missing Flow Duration, fill rest with 0."""
    before = df.count()
    df = df.dropna(subset=["Flow Duration"])
    df = df.fillna(0)
    log.info(f"Missing values handled. Dropped {before - df.count():,} rows.")
    return df


def replace_infinity(df):
    """Replace +inf and -inf in rate columns with 0."""
    for c in ["Flow Bytes/s", "Flow Packets/s"]:
        df = df.withColumn(c,
            when(col(c) == float("inf"),  0)
           .when(col(c) == float("-inf"), 0)
           .otherwise(col(c)))
    log.info("Infinity values replaced.")
    return df


def remove_duplicates(df):
    """Remove fully duplicate rows."""
    before = df.count()
    df = df.dropDuplicates()
    log.info(f"Duplicates removed: {before - df.count():,}")
    return df


def cap_outliers(df, column="Flow Duration"):
    """Cap outliers using IQR method."""
    q1, q3 = df.select(
        percentile_approx(column, 0.25),
        percentile_approx(column, 0.75)
    ).first()
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df.withColumn(column,
        when(col(column) < lower, lower)
       .when(col(column) > upper, upper)
       .otherwise(col(column)))
    log.info(f"Outliers capped. {column} range: [{lower}, {upper}]")
    return df


def encode_labels(df):
    """Add is_attack binary column and label_index multi-class column."""
    df = df.withColumn("Label", trim(col("Label")))
    df = df.withColumn("is_attack",
        when(col("Label") == "BENIGN", 0).otherwise(1))
    indexer = StringIndexer(inputCol="Label", outputCol="label_index")
    df = indexer.fit(df).transform(df)
    log.info("Labels encoded.")
    return df


def engineer_features(df):
    """Add derived feature columns."""
    df = df.withColumn("pkt_byte_ratio",
        when(col("Total Length of Fwd Packets") > 0,
             col("Total Fwd Packets") / col("Total Length of Fwd Packets"))
        .otherwise(0.0))

    df = df.withColumn("fwd_bwd_ratio",
        when(col("Total Backward Packets") > 0,
             col("Total Fwd Packets") / col("Total Backward Packets"))
        .otherwise(col("Total Fwd Packets").cast("float")))

    df = df.withColumn("duration_bucket",
        when(col("Flow Duration") < 1000,    "very_short")
       .when(col("Flow Duration") < 100000,  "short")
       .when(col("Flow Duration") < 1000000, "medium")
       .otherwise("long"))
    log.info("Feature engineering done.")
    return df


def scale_features(df):
    """Assemble numeric features into vector and apply StandardScaler."""
    exclude_cols = ["Label", "label_index", "is_attack",
                    "source_file", "duration_bucket"]
    feature_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, NumericType)
        and f.name not in exclude_cols
    ]
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    df = assembler.transform(df)
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True, withStd=True
    )
    df = scaler.fit(df).transform(df)
    log.info(f"Scaling done. {len(feature_cols)} features assembled.")
    return df


# ── LOAD ──────────────────────────────────────────────────────────

def write_parquet(df, output_dir):
    """Write cleaned data to Parquet partitioned by is_attack."""
    df.drop("features_raw", "features") \
      .repartition(4) \
      .write.mode("overwrite") \
      .partitionBy("is_attack") \
      .parquet(output_dir)
    log.info(f"Parquet written to {output_dir}")


def write_mysql(df, output_dir):
    """Write data to MySQL via JDBC."""
    cols_to_write = [c for c in df.columns
                     if c not in ["features_raw", "features"]]
    df.select(cols_to_write).repartition(4).write.jdbc(
        url="jdbc:mysql://localhost:3306/cicids_db?useSSL=false",
        table="network_traffic",
        mode="overwrite",
        properties={
            "user": "spark_user",
            "password": "password123",
            "driver": "com.mysql.cj.jdbc.Driver"
        }
    )
    log.info("Written to MySQL.")


def write_label_summary(df, output_dir):
    # Write to a sibling folder, NOT inside output_dir
    summary_dir = output_dir.rstrip("/") + "_label_summary/"
    df.groupBy("Label", "is_attack") \
      .count() \
      .orderBy(F.desc("count")) \
      .coalesce(1) \
      .write.mode("overwrite") \
      .csv(summary_dir, header=True)
    log.info(f"Label summary saved to {summary_dir}")



# ── OPTIMIZATION ──────────────────────────────────────────────────

def cleanup(spark):
    """Clear cache and delete temp shuffle files."""
    spark.catalog.clearCache()
    for d in ["/tmp/spark-temp", "./checkpoints/"]:
        if os.path.exists(d):
            shutil.rmtree(d)
            log.info(f"Cleaned: {d}")


def validate_output(spark, output_dir):
    df_check = spark.read.parquet(output_dir)
    log.info(f"Rows written : {df_check.count():,}")
    log.info(f"Columns      : {len(df_check.columns)}")
    df_check.groupBy("is_attack").count().show()
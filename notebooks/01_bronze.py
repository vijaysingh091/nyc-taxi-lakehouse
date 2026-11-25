# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Bronze ingestion (Auto Loader)

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "nyc_taxi"
RAW = f"/Volumes/{CATALOG}/landing/taxi_raw"
TRIPS_PATH = f"{RAW}/trips"
ZONES_PATH = f"{RAW}/zones"
CHECKPOINT = f"{RAW}/_checkpoints/bronze_trips"
BRONZE_TRIPS = f"{CATALOG}.bronze.yellow_trips"
BRONZE_ZONES = f"{CATALOG}.bronze.taxi_zones"

# COMMAND ----------

# Incremental load: the checkpoint remembers files already loaded
bronze_stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "parquet")
    .option("cloudFiles.schemaLocation", CHECKPOINT + "_schema")
    .load(TRIPS_PATH)
    .withColumn("_ingest_ts", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)

(
    bronze_stream.writeStream
    .option("checkpointLocation", CHECKPOINT)
    .trigger(availableNow=True)
    .toTable(BRONZE_TRIPS)
    .awaitTermination()
)

# COMMAND ----------

# Zones: small lookup, full refresh each run
zones = (
    spark.read.csv(ZONES_PATH, header=True, inferSchema=True)
    .withColumn("_ingest_ts", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)
zones.write.format("delta").mode("overwrite").saveAsTable(BRONZE_ZONES)

# COMMAND ----------

print("bronze trips:", spark.table(BRONZE_TRIPS).count())
print("bronze zones:", spark.table(BRONZE_ZONES).count())
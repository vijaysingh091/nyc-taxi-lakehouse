# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Silver: clean, quarantine, enrich

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "nyc_taxi"
BRONZE_TRIPS = f"{CATALOG}.bronze.yellow_trips"
BRONZE_ZONES = f"{CATALOG}.bronze.taxi_zones"
SILVER_TRIPS = f"{CATALOG}.silver.yellow_trips"
SILVER_REJECTED = f"{CATALOG}.silver.yellow_trips_rejected"

# Valid pickup window. Change these when new months are added
START_DATE = "2025-01-01"
END_DATE = "2025-07-01"   # exclusive

KEYS = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID", "total_amount"]

# COMMAND ----------

bronze = spark.table(BRONZE_TRIPS).drop("_rescued_data")
deduped = bronze.dropDuplicates(KEYS)

dur = (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60

reason = (
    F.when((F.col("tpep_pickup_datetime") < START_DATE) |
           (F.col("tpep_pickup_datetime") >= END_DATE), "bad_date")
    .when((dur < 1) | (dur > 180), "bad_duration")
    .when((F.col("trip_distance") <= 0) | (F.col("trip_distance") > 100), "bad_distance")
    .when((F.col("total_amount") <= 0) | (F.col("total_amount") > 500), "bad_total")
    .when((F.col("fare_amount") <= 0) & (F.col("payment_type") != 0), "bad_fare")
)

flagged = deduped.withColumn("reject_reason", reason)

calc = (
    F.col("fare_amount") + F.col("extra") + F.col("mta_tax") + F.col("tip_amount")
    + F.col("tolls_amount") + F.col("improvement_surcharge")
    + F.coalesce(F.col("congestion_surcharge"), F.lit(0.0))
    + F.coalesce(F.col("Airport_fee"), F.lit(0.0))
    + F.col("cbd_congestion_fee")
)

# COMMAND ----------

# Rejected rows go to quarantine with a reason
(flagged.filter(F.col("reject_reason").isNotNull())
    .write.format("delta").mode("overwrite").saveAsTable(SILVER_REJECTED))

# COMMAND ----------

zones = spark.table(BRONZE_ZONES)
pu = zones.select(F.col("LocationID").alias("PULocationID"),
                  F.col("Borough").alias("pickup_borough"),
                  F.col("Zone").alias("pickup_zone"))
do = zones.select(F.col("LocationID").alias("DOLocationID"),
                  F.col("Borough").alias("dropoff_borough"),
                  F.col("Zone").alias("dropoff_zone"))

silver = (
    flagged.filter(F.col("reject_reason").isNull()).drop("reject_reason")
    .withColumn("passenger_count",
                F.when(F.col("passenger_count") == 0, None)
                 .otherwise(F.col("passenger_count").cast("int")))
    .withColumn("RatecodeID", F.col("RatecodeID").cast("int"))
    .withColumn("payment_type", F.col("payment_type").cast("int"))
    .withColumn("trip_duration_min", F.round(dur, 2))
    .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .withColumn("is_flex_fare", F.col("payment_type") == 0)
    .withColumn("fees_match_total", F.abs(calc - F.col("total_amount")) < 0.02)
    .join(F.broadcast(pu), "PULocationID", "left")
    .join(F.broadcast(do), "DOLocationID", "left")
)

silver.write.format("delta").mode("overwrite").saveAsTable(SILVER_TRIPS)

# COMMAND ----------

# Reconciliation: nothing may be lost
d = deduped.count()
s = spark.table(SILVER_TRIPS).count()
r = spark.table(SILVER_REJECTED).count()
print("after dedupe:", d, "| silver:", s, "| rejected:", r)
assert s + r == d, "silver + rejected must equal deduped bronze"
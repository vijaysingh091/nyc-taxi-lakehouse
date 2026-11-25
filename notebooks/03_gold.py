# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Gold: business tables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "nyc_taxi"
SILVER_TRIPS = f"{CATALOG}.silver.yellow_trips"
GOLD_DAILY = f"{CATALOG}.gold.daily_revenue"
GOLD_HOURLY = f"{CATALOG}.gold.hourly_demand"
GOLD_ZONES = f"{CATALOG}.gold.top_pickup_zones"
GOLD_PAYMENT = f"{CATALOG}.gold.payment_summary"

s = spark.table(SILVER_TRIPS)

# COMMAND ----------

daily = (
    s.groupBy("pickup_date")
    .agg(
        F.count("*").alias("trips"),
        F.round(F.sum("total_amount"), 2).alias("revenue"),
        F.round(F.avg("total_amount"), 2).alias("avg_total"),
        F.round(F.avg("trip_distance"), 2).alias("avg_distance_mi"),
        F.round(F.avg("trip_duration_min"), 2).alias("avg_duration_min"),
    )
)
w7 = Window.orderBy("pickup_date").rowsBetween(-6, 0)
daily = daily.withColumn("revenue_7d_avg", F.round(F.avg("revenue").over(w7), 2))
daily.write.format("delta").mode("overwrite").saveAsTable(GOLD_DAILY)

# COMMAND ----------

hourly = (
    s.withColumn("dow_num", F.dayofweek("tpep_pickup_datetime"))
     .withColumn("dow", F.date_format("tpep_pickup_datetime", "E"))
     .groupBy("dow_num", "dow", "pickup_hour")
     .agg(
         F.count("*").alias("trips"),
         F.round(F.avg("total_amount"), 2).alias("avg_total"),
     )
)
hourly.write.format("delta").mode("overwrite").saveAsTable(GOLD_HOURLY)

# COMMAND ----------

zones_agg = (
    s.groupBy("PULocationID", "pickup_borough", "pickup_zone")
    .agg(
        F.count("*").alias("trips"),
        F.round(F.sum("total_amount"), 2).alias("revenue"),
        F.round(F.avg("total_amount"), 2).alias("avg_total"),
        F.round(F.avg("trip_distance"), 2).alias("avg_distance_mi"),
    )
)
w_all = Window.orderBy(F.desc("trips"))
w_boro = Window.partitionBy("pickup_borough").orderBy(F.desc("trips"))
zones_agg = (
    zones_agg
    .withColumn("rank_overall", F.dense_rank().over(w_all))
    .withColumn("rank_in_borough", F.dense_rank().over(w_boro))
)
zones_agg.write.format("delta").mode("overwrite").saveAsTable(GOLD_ZONES)

# COMMAND ----------

names = {0: "Flex Fare", 1: "Credit card", 2: "Cash",
         3: "No charge", 4: "Dispute", 5: "Unknown", 6: "Voided trip"}
name_map = F.create_map([F.lit(x) for kv in names.items() for x in kv])

pay = (
    s.groupBy("payment_type")
    .agg(
        F.count("*").alias("trips"),
        F.round(F.sum("total_amount"), 2).alias("revenue"),
        F.round(F.avg("total_amount"), 2).alias("avg_total"),
        F.round(F.avg("tip_amount"), 2).alias("avg_tip"),
        F.round(100 * F.sum("tip_amount") / F.sum("fare_amount"), 2).alias("tip_pct_of_fare"),
    )
    .withColumn("payment_name", name_map[F.col("payment_type")])
    .withColumn("share_pct", F.round(100 * F.col("trips") / F.sum("trips").over(Window.partitionBy()), 2))
    .select("payment_type", "payment_name", "trips", "share_pct",
            "revenue", "avg_total", "avg_tip", "tip_pct_of_fare")
)
pay.write.format("delta").mode("overwrite").saveAsTable(GOLD_PAYMENT)

# COMMAND ----------

# Reconciliation: every Gold table must add up to Silver
silver_n = s.count()
for t in [GOLD_DAILY, GOLD_HOURLY, GOLD_ZONES, GOLD_PAYMENT]:
    n = spark.table(t).agg(F.sum("trips")).first()[0]
    print(t, n)
    assert n == silver_n, f"{t}: {n} != {silver_n}"
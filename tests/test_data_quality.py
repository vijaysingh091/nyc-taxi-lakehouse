import pytest
from pyspark.sql import functions as F

CATALOG = "nyc_taxi"
BRONZE = f"{CATALOG}.bronze.yellow_trips"
SILVER = f"{CATALOG}.silver.yellow_trips"
REJECTED = f"{CATALOG}.silver.yellow_trips_rejected"
G_DAILY = f"{CATALOG}.gold.daily_revenue"
G_HOURLY = f"{CATALOG}.gold.hourly_demand"
G_ZONES = f"{CATALOG}.gold.top_pickup_zones"
G_PAYMENT = f"{CATALOG}.gold.payment_summary"
G_METRICS = f"{CATALOG}.gold.model_metrics"

KEYS = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID", "total_amount"]
START_DATE = "2025-01-01"
END_DATE = "2025-07-01"
KNOWN_REASONS = {"bad_date", "bad_duration", "bad_distance", "bad_total", "bad_fare"}


def test_bronze_has_no_rescued_data(spark):
    bad = spark.table(BRONZE).filter(F.col("_rescued_data").isNotNull()).count()
    assert bad == 0


def test_silver_plus_rejected_equals_deduped_bronze(spark):
    deduped = spark.table(BRONZE).drop("_rescued_data").dropDuplicates(KEYS).count()
    total = spark.table(SILVER).count() + spark.table(REJECTED).count()
    assert total == deduped


def test_silver_has_no_duplicates(spark):
    s = spark.table(SILVER)
    assert s.count() == s.dropDuplicates(KEYS).count()


def test_silver_value_ranges(spark):
    r = spark.table(SILVER).agg(
        F.min("trip_distance").alias("min_dist"),
        F.max("trip_distance").alias("max_dist"),
        F.min("total_amount").alias("min_total"),
        F.max("total_amount").alias("max_total"),
        F.min("trip_duration_min").alias("min_dur"),
        F.max("trip_duration_min").alias("max_dur"),
    ).first()
    assert r.min_dist > 0 and r.max_dist <= 100
    assert r.min_total > 0 and r.max_total <= 500
    assert r.min_dur >= 1 and r.max_dur <= 180


def test_silver_dates_in_window(spark):
    r = spark.table(SILVER).agg(
        F.min("tpep_pickup_datetime").alias("lo"),
        F.max("tpep_pickup_datetime").alias("hi"),
    ).first()
    assert str(r.lo) >= START_DATE
    assert str(r.hi) < END_DATE


def test_silver_zone_names_present(spark):
    bad = spark.table(SILVER).filter(
        F.col("pickup_zone").isNull() | F.col("dropoff_zone").isNull()).count()
    assert bad == 0


def test_rejected_reasons_are_known(spark):
    reasons = {r[0] for r in spark.table(REJECTED).select("reject_reason").distinct().collect()}
    assert reasons <= KNOWN_REASONS


@pytest.mark.parametrize("table", [G_DAILY, G_HOURLY, G_ZONES, G_PAYMENT])
def test_gold_trips_match_silver(spark, table):
    silver_n = spark.table(SILVER).count()
    gold_n = spark.table(table).agg(F.sum("trips")).first()[0]
    assert gold_n == silver_n


def test_gold_daily_days_match_silver(spark):
    days = spark.table(SILVER).select("pickup_date").distinct().count()
    assert spark.table(G_DAILY).count() == days


def test_gold_hourly_has_at_most_168_rows(spark):
    assert spark.table(G_HOURLY).count() <= 168


def test_payment_shares_sum_to_100(spark):
    total = spark.table(G_PAYMENT).agg(F.sum("share_pct")).first()[0]
    assert abs(total - 100) < 0.1


def test_gbt_beats_baseline(spark):
    rows = {r["model"]: r["rmse"] for r in spark.table(G_METRICS).collect()}
    assert rows["gbt_regressor"] < rows["baseline_mean"]
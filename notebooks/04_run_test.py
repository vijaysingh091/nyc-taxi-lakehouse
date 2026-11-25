# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Run data quality tests (pytest)

# COMMAND ----------

# MAGIC %pip install pytest

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import pytest

sys.dont_write_bytecode = True

# This notebook lives in <repo>/notebooks, tests live in <repo>/tests
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
tests_dir = os.path.join(repo_root, "tests")
print("tests dir:", tests_dir)

code = pytest.main(["-v", "-p", "no:cacheprovider", tests_dir])
assert code == 0, f"pytest failed with exit code {code}"
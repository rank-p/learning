import time
from microspark import SparkContext as MicroSparkContext
from pyspark.sql import SparkSession

with open("pride_and_prejudice.txt") as f:
    lines = f.readlines()

print(f"Dataset: {len(lines)} lines, {sum(len(l.split()) for l in lines)} words")
print()

num_partitions = 4

# ---------------------------------------------------------------------------
# Microspark
# ---------------------------------------------------------------------------
print("=== Microspark ===")
sc = MicroSparkContext(max_workers=num_partitions)

t = time.perf_counter()
micro_result = sc.parallelize(lines, num_partitions) \
    .flatMap(str.split) \
    .map(lambda w: (w.lower(), 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .collect()
micro_time = time.perf_counter() - t

micro_dict = dict(micro_result)
micro_top10 = sorted(micro_result, key=lambda x: x[1], reverse=True)[:10]
print(f"Time: {micro_time:.3f}s")
print(f"Unique words: {len(micro_result)}")
print(f"Top 10: {micro_top10}")
print()

# ---------------------------------------------------------------------------
# PySpark
# ---------------------------------------------------------------------------
print("=== PySpark ===")
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("benchmark") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
psc = spark.sparkContext

t = time.perf_counter()
py_result = psc.parallelize(lines, num_partitions) \
    .flatMap(str.split) \
    .map(lambda w: (w.lower(), 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .collect()
py_time = time.perf_counter() - t

py_dict = dict(py_result)
py_top10 = sorted(py_result, key=lambda x: x[1], reverse=True)[:10]
print(f"Time: {py_time:.3f}s")
print(f"Unique words: {len(py_result)}")
print(f"Top 10: {py_top10}")

spark.stop()

# ---------------------------------------------------------------------------
# Correctness comparison
# ---------------------------------------------------------------------------
print()
print("=== Correctness ===")
match = micro_dict == py_dict
print(f"Results match: {match}")
if not match:
    only_micro = set(micro_dict) - set(py_dict)
    only_py = set(py_dict) - set(micro_dict)
    diff_counts = {k for k in micro_dict if k in py_dict and micro_dict[k] != py_dict[k]}
    if only_micro: print(f"  Only in microspark: {len(only_micro)} words")
    if only_py: print(f"  Only in pyspark: {len(only_py)} words")
    if diff_counts: print(f"  Different counts: {len(diff_counts)} words")

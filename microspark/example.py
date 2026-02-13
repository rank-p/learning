from microspark import SparkContext

sc = SparkContext()
assert sc.parallelize([1,2,3,4,5,6], 3).collect() == [1,2,3,4,5,6]
assert sc.parallelize([1,2,3,4,5,6], 4).collect() == [1,2,3,4,5,6]
assert sc.parallelize([], 2).collect() == []
print("Step 1 passed!")

# Step 2: map, filter, flatMap
assert sc.parallelize(range(10), 2).map(lambda x: x * 2).filter(lambda x: x > 10).collect() == [12, 14, 16, 18]
assert sc.parallelize(["hello world", "foo bar baz"], 2).flatMap(lambda s: s.split()).collect() == ["hello", "world", "foo", "bar", "baz"]

# Verify laziness: transformed RDD has no _partitions of its own
lazy = sc.parallelize(range(5), 1).map(lambda x: x + 1)
assert lazy._partitions == []

print("Step 2 passed!")

rdd = sc.parallelize(range(1, 11), 3)
assert rdd.count() == 10
assert rdd.reduce(lambda a, b: a + b) == 55
assert rdd.take(3) == [1, 2, 3]
assert len(rdd.take(100)) == 10  # asking for more than exists
assert rdd.reduce(lambda a, b: a * b) == 3628800  # 10! — verifies no hardcoded 0
assert sc.parallelize(range(10), 3).map(lambda x: x * 2).count() == 10  # count after transform
assert sc.parallelize([1], 1).take(3) == [1]  # take more than exists
print("Step 3 passed!")

# ---------------------------------------------------------------------------
# Step 4: Key-value RDDs — mapValues, keys, values
# ---------------------------------------------------------------------------
pairs = sc.parallelize([(1, "a"), (2, "b"), (3, "c")], 2)
assert pairs.mapValues(str.upper).collect() == [(1, "A"), (2, "B"), (3, "C")]
assert pairs.keys().collect() == [1, 2, 3]
assert pairs.values().collect() == ["a", "b", "c"]
assert sc.parallelize([(1, 10), (2, 20)], 1).mapValues(lambda v: v + 1).collect() == [(1, 11), (2, 21)]
print("Step 4 passed!")

# ---------------------------------------------------------------------------
# Step 5: Shuffle — groupByKey, reduceByKey
# ---------------------------------------------------------------------------
# Word count
wc = sc.parallelize(["hello world", "hello spark"], 2) \
       .flatMap(str.split) \
       .map(lambda w: (w, 1)) \
       .reduceByKey(lambda a, b: a + b)
wc_result = dict(wc.collect())
assert wc_result == {"hello": 2, "world": 1, "spark": 1}

# groupByKey
gk = sc.parallelize([(1, "a"), (1, "b"), (2, "c")], 2).groupByKey()
gk_result = {k: sorted(v) for k, v in gk.collect()}
assert gk_result == {1: ["a", "b"], 2: ["c"]}

# reduceByKey with multiplication
rk = sc.parallelize([(1, 2), (1, 3), (2, 5), (2, 4)], 2).reduceByKey(lambda a, b: a * b)
rk_result = dict(rk.collect())
assert rk_result == {1: 6, 2: 20}

print("Step 5 passed!")

# ---------------------------------------------------------------------------
# Step 6: DAG visualization + stage splitting
# ---------------------------------------------------------------------------
dag_rdd = sc.parallelize(["hello world", "hello spark"], 2) \
             .flatMap(str.split) \
             .map(lambda w: (w, 1)) \
             .reduceByKey(lambda a, b: a + b) \
             .mapValues(lambda v: v * 10)

debug = dag_rdd.toDebugString()
print("\nDAG debug string:")
print(debug)

# Should have 2 stages: narrows before shuffle, then shuffle + narrows after
assert "Stage 0" in debug
assert "Stage 1" in debug

dag_rdd.draw_dag().render('dag', view=True)
print("DAG graph rendered to dag.svg")

print("Step 6 passed!")

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


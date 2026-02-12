from microspark import SparkContext

sc = SparkContext()
assert sc.parallelize([1,2,3,4,5,6], 3).collect() == [1,2,3,4,5,6]
assert sc.parallelize([1,2,3,4,5,6], 4).collect() == [1,2,3,4,5,6]
assert sc.parallelize([], 2).collect() == []
print("Step 1 passed!")


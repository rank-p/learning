import time
from microspark import SparkContext

with open("pride_and_prejudice.txt") as f:
    lines = f.readlines()

print(f"Dataset: {len(lines)} lines, {sum(len(l.split()) for l in lines)} words")

num_partitions = 2
sc = SparkContext(max_workers=num_partitions)

# Build pipeline step by step, timing each action
rdd = sc.parallelize(lines, num_partitions)

t = time.perf_counter()
rdd_flat = rdd.flatMap(str.split)
print(f"[{time.perf_counter()-t:.3f}s] flatMap built (lazy)")

t = time.perf_counter()
rdd_pairs = rdd_flat.map(lambda w: (w.lower(), 1))
print(f"[{time.perf_counter()-t:.3f}s] map built (lazy)")

# Force compute before shuffle to see narrow cost
t = time.perf_counter()
pair_count = rdd_pairs.count()
print(f"[{time.perf_counter()-t:.3f}s] count (forces narrow transforms): {pair_count} pairs")

# Now the expensive part: reduceByKey (shuffle)
t = time.perf_counter()
rdd_reduced = rdd_pairs.reduceByKey(lambda a, b: a + b)
print(f"[{time.perf_counter()-t:.3f}s] reduceByKey built (lazy)")

t = time.perf_counter()
result = rdd_reduced.collect()
print(f"[{time.perf_counter()-t:.3f}s] collect (forces shuffle + reduce): {len(result)} unique words")

# Show top 10
top10 = sorted(result, key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 words: {top10}")

# Microspark — Build Plan

Minimal Apache Spark from scratch. Karpathy's micrograd, but for distributed data processing.

## Steps

- [x] **Step 1: Partition + RDD + collect()**
  `Partition` wrapper, `RDD` with partitions, `SparkContext.parallelize()`, `collect()`

- [x] **Step 2: Narrow transformations — map, filter, flatMap**
  Lazy DAG construction, parent/transform chain, nothing runs until an action

- [x] **Step 3: More actions — count, reduce, take**
  Different ways to trigger computation and aggregate results

- [ ] **Step 4: Key-value RDDs — mapValues, keys, values**
  Pair RDDs: `(key, value)` tuples. Foundation for shuffle operations

- [ ] **Step 5: Shuffle — groupByKey, reduceByKey**
  Wide dependencies, `HashPartitioner`, shuffle write/read, map-side pre-aggregation

- [ ] **Step 6: DAG visualization + stage splitting**
  `toDebugString()`, walk lineage, split stages at shuffle boundaries

- [ ] **Step 7: Parallel execution**
  `ThreadPoolExecutor`, partitions compute in parallel within a stage

- [ ] **Step 8: Fault tolerance via lineage**
  Cache partitions, simulate failure, recompute from lineage

- [ ] **Step 9: Word count end-to-end**
  Full pipeline on a text file, compare with real PySpark API

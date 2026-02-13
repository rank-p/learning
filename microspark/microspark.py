from collections import defaultdict
from graphviz import Digraph

class Partition:
    def __init__(self, index: int, data):
        self._index = index
        self.data = list(data)


class RDD:
    def __init__(self, context, partitions=None, parent=None, transform_fn=None, name=None):
        self._context = context
        self._partitions: list = partitions or []
        self._parent = parent
        self._transform_fn = transform_fn
        self._name = name

    def compute(self, partition_idx: int) -> Partition:
        if self._partitions:
            return self._partitions[partition_idx]
        else:
            assert self._transform_fn is not None
            assert self._parent is not None
            return self._transform_fn(self._parent.compute(partition_idx))

    def map(self, map_fn) -> RDD:
        def _transform(p: Partition) -> Partition:
            return Partition(p._index, [map_fn(item) for item in p.data])
        return RDD(self._context, parent=self, transform_fn=_transform, name="map")

    def flatMap(self, flatmap_fn) -> RDD:
        def _transform(p: Partition) -> Partition:
            res = []
            for item in p.data:
                res.extend(flatmap_fn(item))
            return Partition(p._index, res)
        return RDD(self._context, parent=self, transform_fn=_transform, name="flatMap")

    def filter(self, filter_fn) -> RDD:
        def _transform(p: Partition) -> Partition:
            return Partition(p._index, [item for item in p.data if filter_fn(item)])
        return RDD(self._context, parent=self, transform_fn=_transform, name="filter")

    def reduce(self, reduce_fn):
        # reduce within partitions
        partials = []
        for i in range(self.num_partitions):
            data = self.compute(i).data
            if not data:
                continue
            acc = data[0]
            for item in data[1:]:
                acc = reduce_fn(acc, item)
            partials.append(acc)

        if not partials:
            raise ValueError("cannot reduce an empty RDD")
        
        # reduce accross partitions
        acc = partials[0]
        for v in partials[1:]:
            acc = reduce_fn(acc, v)
        return acc

    def count(self) -> int:
        return sum(len(self.compute(i).data) for i in range(self.num_partitions))

    def take(self, n: int):
        res = []
        for i in range(self.num_partitions):
            if len(res) >= n:
                break
            res.extend(self.compute(i).data)
        return res

    def collect(self) -> list:
        result = []
        for i in range(self.num_partitions):
            result.extend(self.compute(i).data)
        return result

    def mapValues(self, map_fn):
        # assuming data is of type list((key,value))
        rdd = self.map(lambda pair: (pair[0], map_fn(pair[1])))
        rdd._name = "mapValues"
        return rdd

    def keys(self):
        rdd = self.map(lambda x: x[0])
        rdd._name = "keys"
        return rdd

    def values(self):
        rdd = self.map(lambda x: x[1])
        rdd._name = "values"
        return rdd

    def groupByKey(self) -> RDD:
        shuffled_rdd = ShuffledRDD(self._context, parent=self)
        def _group(p: Partition):
            groups = defaultdict(list)
            for key, value in p.data:
                groups[key].append(value)
            return Partition(p._index, list(groups.items()))
        return RDD(self._context, parent=shuffled_rdd, transform_fn=_group, name="groupByKey")
    
    def reduceByKey(self, reduce_fn) -> RDD:
        def _reduce(p: Partition):
            groups = {}
            for (key, value) in p.data:
                if key not in groups:
                    groups[key] = value
                else:
                    groups[key] = reduce_fn(groups[key], value)
            return Partition(p._index, list(groups.items()))
        # pre-reduce in partitions
        combined = RDD(self._context, parent=self, transform_fn=_reduce, name="reduceByKey")
        shuffled = ShuffledRDD(self._context, parent=combined)
        return RDD(self._context, parent=shuffled, transform_fn=_reduce, name="reduceByKey")
            
            

    def toDebugString(self) -> str:
        stage_lists = self.stages()
        lines = []
        for stage_idx, stage in enumerate(stage_lists):
            lines.append(f"Stage {stage_idx}:")
            for name in stage:
                lines.append(f"  ({self.num_partitions}) {name}")
        return "\n".join(lines)

    def stages(self):
        stages = [[]]
        rdd = self
        while rdd is not None:
            stages[-1].append(rdd._name or "RDD")
            if isinstance(rdd, ShuffledRDD):
                stages.append([])
            rdd = rdd._parent
        return stages

    def draw_dag(self):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        # collect all RDDs in lineage
        rdds = []
        rdd = self
        while rdd is not None:
            rdds.append(rdd)
            rdd = rdd._parent
        # draw each RDD as a cluster containing partition nodes
        for rdd in rdds:
            uid = str(id(rdd))
            name = rdd._name or "RDD"
            color = "salmon" if isinstance(rdd, ShuffledRDD) else "lightblue"
            with dot.subgraph(name=f"cluster_{uid}") as sub:
                sub.attr(label=name, style="rounded,filled", fillcolor=color)
                for p in range(rdd.num_partitions):
                    sub.node(f"{uid}_p{p}", label=f"P{p}", shape="box",
                             style="filled", fillcolor="white", width="0.4", height="0.3")
        # draw edges between partitions
        for rdd in rdds:
            if not rdd._parent:
                continue
            uid = str(id(rdd))
            parent_uid = str(id(rdd._parent))
            if isinstance(rdd, ShuffledRDD):
                # wide: every parent partition → every child partition
                for pp in range(rdd._parent.num_partitions):
                    for cp in range(rdd.num_partitions):
                        dot.edge(f"{parent_uid}_p{pp}", f"{uid}_p{cp}", color="red")
            else:
                # narrow: partition i → partition i
                for p in range(rdd.num_partitions):
                    dot.edge(f"{parent_uid}_p{p}", f"{uid}_p{p}")
        return dot

    @property
    def num_partitions(self) -> int:
        if self._partitions:
            return len(self._partitions)
        if self._parent:
            return self._parent.num_partitions
        raise ValueError("Illegal state, neither _partitions nor _parent defined")

class ShuffledRDD(RDD):

    def __init__(self, context, parent: RDD, num_partitions=None):
        assert parent is not None
        super().__init__(context=context, partitions=None, parent=parent, transform_fn=None, name="ShuffledRDD")
        # If num_partitions is passed use that (repartition) or fallback to parent
        n = num_partitions or parent.num_partitions
        self.partitioner = HashPartitioner(num_partitions=n)

    def compute(self, partition_idx: int) -> Partition:
        if self._partitions:
            return self._partitions[partition_idx]
        
        assert self._parent is not None
        shuffled_data = []
        for i in range(self.num_partitions):
            p = self._parent.compute(i)
            for (key, value) in p.data:
                if self.partitioner.partition(key) == partition_idx:
                    shuffled_data.append((key, value))
        return Partition(partition_idx, shuffled_data)

    @property
    def num_partitions(self):
        return self.partitioner.num_partitions
                





class SparkContext:

    def __init__(self):
        pass

    def parallelize(self, data, num_partitions: int) -> RDD:
        if num_partitions <= 0:
            return RDD(SparkContext(), [])

        chunk_size = len(data) // num_partitions
        partitions = []
        for i in range(num_partitions):
            start = i * chunk_size
            if i == num_partitions - 1:
                end = len(data)
            else:
                end = start + chunk_size
            partitions.append(Partition(i, data[start:end]))
        return RDD(self, partitions, name="parallelize")

class HashPartitioner:
    
    def __init__(self, num_partitions):
        self.num_partitions = num_partitions

    def partition(self, key):
        # % function always returns positive results in python (hash can be negative)
        return hash(key) % self.num_partitions


from collections import defaultdict

class Partition:
    def __init__(self, index: int, data):
        self._index = index
        self.data = list(data)


class RDD:
    def __init__(self, context, partitions=None, parent=None, transform_fn=None):
        self._context = context
        self._partitions: list = partitions or []
        self._parent = parent
        self._transform_fn = transform_fn

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
        return RDD(self._context, parent=self, transform_fn=_transform)

    def flatMap(self, flatmap_fn) -> RDD:
        def _transform(p: Partition) -> Partition:
            res = []
            for item in p.data:
                res.extend(flatmap_fn(item))
            return Partition(p._index, res)
        return RDD(self._context, parent=self, transform_fn=_transform)

    def filter(self, filter_fn) -> RDD:
        def _transform(p: Partition) -> Partition:
            return Partition(p._index, [item for item in p.data if filter_fn(item)])
        return RDD(self._context, parent=self, transform_fn=_transform)

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
        return self.map(lambda pair: (pair[0], map_fn(pair[1])))

    def keys(self):
        return self.map(lambda x: x[0])

    def values(self):
        return self.map(lambda x: x[1])

    def groupByKey(self) -> RDD:
        shuffled_rdd = ShuffledRDD(self._context, parent=self)
        def _group(p: Partition):
            groups = defaultdict(list)
            for key, value in p.data:
                groups[key].append(value)
            return Partition(p._index, list(groups.items()))
        return RDD(self._context, parent=shuffled_rdd, transform_fn=_group)
    
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
        combined = RDD(self._context, parent=self, transform_fn=_reduce)
        shuffled = ShuffledRDD(self._context, parent=combined)
        return RDD(self._context, parent=shuffled, transform_fn=_reduce)
            
            

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
        super().__init__(context=context, partitions=None, parent=parent, transform_fn=None)
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
        return RDD(self, partitions)

class HashPartitioner:
    
    def __init__(self, num_partitions):
        self.num_partitions = num_partitions

    def partition(self, key):
        # % function always returns positive results in python (hash can be negative)
        return hash(key) % self.num_partitions


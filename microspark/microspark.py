
class Partition:
    def __init__(self, index: int, data):
        self._index = index
        self.data = list(data)


class RDD:
    def __init__(self, context, partitions=None):
        self._context = context
        self._partitions: list = partitions or []

    def collect(self):
        return [item for p in self._partitions for item in p.data]

    @property
    def num_partitions(self) -> int:
        return len(self._partitions)


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



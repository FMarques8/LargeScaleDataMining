# Bucket and DGIM class file
# Created for MLSD Assignment 3 - Structured Streaming - DGIM

class Bucket:
    def __init__(self, timestamp, size):
        self.timestamp = timestamp
        self.size = size

class DGIM:
    def __init__(self, window_size, k):
        self.window_size = window_size
        self.k = k
        self.stream_timestamp = 0
        self.buckets = []
        self.real_count = 0
        self.estimated_count = 0

    def update(self, bit):
        self.stream_timestamp += 1
        
        if int(bit) == 1:
            self.real_count += 1
            self.buckets.insert(0, Bucket(self.stream_timestamp, 1))
            self._merge_buckets()
        if len(self.buckets) > 2:
            self._adjust_last_bucket()
            self._evict_old_buckets()

    def count(self):
        return sum([b.size for b in self.buckets])
    
    def _merge_buckets(self):
        if len(self.buckets) > 2:
            for i in range(len(self.buckets) - 2):
                if self.buckets[i].size == self.buckets[i+1].size == self.buckets[i+2].size:
                    self.buckets[i+2].size *= 2
                    del self.buckets[i]
                    self._adjust_last_bucket()
                    break

    def _evict_old_buckets(self):
        if self.buckets[-1].timestamp <= self.stream_timestamp - self.window_size:
            del self.buckets[-1]

    def _adjust_last_bucket(self):
        if self.buckets:
            last_bucket = self.buckets[-1]
            if last_bucket.timestamp <= self.stream_timestamp - self.k + 1:
                last_bucket.size = self.buckets[-2].size // 2
"""Emit items in priority order."""

import heapq


class PriorityQueue():

    def __init__(self):
        self.heap = []
        self.next_tie_breaker = 0

    def push(self, priority, value):

        heapq.heappush(
            self.heap,
            (priority, self.next_tie_breaker, value))

        self.next_tie_breaker += 1

    def pop(self):
        priority, _, value = heapq.heappop(self.heap)
        return priority, value

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return ("<PriorityQueue:: len:{}>".format(len(self.heap)))

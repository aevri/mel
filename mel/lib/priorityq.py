"""Emit items in priority order."""

import heapq


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.next_tie_breaker = 0

    def push(self, priority, value):

        heapq.heappush(self.heap, (priority, self.next_tie_breaker, value))

        self.next_tie_breaker += 1

    def pop(self):
        priority, _, value = heapq.heappop(self.heap)
        return priority, value

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return "<PriorityQueue:: len:{}>".format(len(self.heap))


# -----------------------------------------------------------------------------
# Copyright (C) 2017 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------

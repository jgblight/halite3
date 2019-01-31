import heapq

from player.utils import Timer, log_message
from player.constants import DIRECTION_ORDER

class Elem:

    def __init__(self, position, distance, previous):
        self.position = position
        self.distance = distance
        self.previous = previous
        self.removed = False

    def __eq__(self, other):
        return isinstance(other, Elem) and self.position == other.position

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.position.x == other.position.x:
            return self.position.y < other.position.y
        return self.position.x < other.position.x

    def __gt__(self, other):
        if self.position.x == other.position.x:
            return self.position.y > other.position.y
        return self.position.x > other.position.x

    def __le__(self, other):
        return not self.__gt__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __hash__(self):
        return self.position.__hash__()


class Graph:

    def __init__(self, dest):
        self.dest = dest
        self.pq = []
        self.queue_finder = {}
        self.elemmap = {}

    def queue_push(self, elem):
        f_score = 1000 * (abs(elem.position.x - self.dest.x) + abs(elem.position.y - self.dest.y))
        t = (elem.distance + f_score, elem)
        self.queue_finder[elem.position] = t
        heapq.heappush(self.pq, t)

    def queue_remove(self, position):
        t = self.queue_finder.get(position)
        if t:
            t[-1].removed = True

    def minpop(self):
        while self.pq:
            t = heapq.heappop(self.pq)
            if not t[-1].removed:
                elem = t[-1]
                del self.queue_finder[elem.position]
                return elem

    def has(self, position):
        return position in self.elemmap

    def queue_empty(self):
        return len(self.pq) == 0

    def update_min_path(self, position, distance, previous):
        e = self.elemmap.get(position)
        if e and distance >= e.distance:
            return
        if not e: # new entry
            e = Elem(position, distance, previous)
            self.elemmap[position] = e
            self.queue_push(e)
        else: # update entry
            e.distance = distance
            e.previous = previous
            self.queue_remove(position)
            self.queue_push(e)

    def find_first(self, source, destination):
        current = self.elemmap[destination]
        while current.previous.position != source:
            current = self.elemmap[current.previous.position]
        return current.position

def is_safe(game_map, suitor, me):
    if game_map[suitor].is_occupied:
        return False
    for delta in DIRECTION_ORDER:
        neighbour = game_map[suitor.directional_offset(delta)]
        if neighbour.is_occupied and neighbour.ship.owner != me:
            return False
    return True


def djikstra_traverse(game_map, source, destination):
    pq = Graph(destination.position)

    pq.update_min_path(source.position, 0, None)

    me = source.ship.owner

    with Timer("navigation"):
        while not pq.queue_empty():
            current = pq.minpop()
            if current is None:
                return None

            for suitor in current.position.get_surrounding_cardinals():
                suitor = game_map.normalize(suitor)
                if suitor == destination.position or is_safe(game_map, suitor, me):
                    new_dist = current.distance + game_map[suitor].halite_amount + 500
                    pq.update_min_path(suitor, new_dist, current)

            if pq.has(destination.position):
                break

    if not pq.has(destination.position):
        return None

    next_step = pq.find_first(source.position, destination.position)
    return game_map[next_step]

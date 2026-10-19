from datetime import datetime


class TemporalInterval:
    def __init__(self, granularity: str, interval_start: int, interval_end: int, nr_deltas: int):
        self.granularity: str = granularity
        self.interval_start: int = interval_start
        self.interval_end: int = interval_end
        self.nr_deltas = nr_deltas
        #
        # calculates str of intervals
        date_interval_start = datetime.fromtimestamp(self.interval_start)
        date_interval_start_str = date_interval_start.strftime('%Y.%m.%d')
        self.date_interval_start_str = date_interval_start_str
        #
        date_interval_end = datetime.fromtimestamp(self.interval_end)
        date_interval_end_str = date_interval_end.strftime('%Y.%m.%d')
        self.date_interval_end_str = date_interval_end_str


    def __hash__(self):
        # Combine the hashes of the attributes to create a unique hash for the instance
        return hash((self.granularity, self.interval_start, self.interval_end, self.nr_deltas))

    def __eq__(self, other):
        # Compare the attributes for equality
        return (self.granularity, self.interval_start, self.interval_end, self.nr_deltas) == \
            (other.granularity, other.interval_start, other.interval_end, other.nr_deltas)

    def __str__(self):
        # Return a string representation of the instance
        return (f'TemporalInterval('
                f'granularity={self.granularity}, '
                f'interval_start={self.interval_start}, '
                f'interval_end={self.interval_end}, '
                f'nr_deltas={self.nr_deltas})')

    def __repr__(self):
        return self.__str__()

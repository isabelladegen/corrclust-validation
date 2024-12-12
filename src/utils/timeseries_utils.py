from dataclasses import dataclass, field


@dataclass
class TimeColumns:  # for additional time features
    day_of_year = "day of year"
    week_of_year = "week of year"
    month = 'months'
    year = 'years'
    week_day = 'weekdays'
    hour = 'hours'


@dataclass
class Segmentation:
    """ Describes segment length of TS

    """
    length: int = 0  # max length if variable length allowed
    description: str = ''
    x_ticks: [int] = field(default_factory=lambda: [])
    name: str = 'Base'


@dataclass
class DailyTimeseries(Segmentation):
    """ Class to describe daily time series
    - Df consists of hourly readings per day, max 24 readings
    - each day is a new time series
    """
    length: int = 24
    description: str = 'Hour of day (UTC)'
    x_ticks: [int] = field(default_factory=lambda: list(range(0, 24, 2)))
    name = 'Days'

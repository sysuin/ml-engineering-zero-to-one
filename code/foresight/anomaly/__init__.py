"""Foresight's anomaly alerts: daily ticket counts by supplier and by
category, watched by a Poisson control chart and an isolation forest,
with each alert carrying its evidence (Chapter 18).

    counts.py    the daily series
    control.py   control charts, sigma and Poisson
    forest.py    isolation forests and path lengths
    alert.py     an alert, its evidence, and how detectors are judged
    job.py       the daily job: python -m foresight.anomaly.job --day D
"""

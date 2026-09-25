"""Foresight's monitoring: what the deployed models see, what they say,
what happens to what they said, and when to replace them (Chapter 24).

    cohorts.py    every contract past its mark, labelled only once its
                  outcome is on record; notices; exposure to a price
                  rise
    psi.py        the population stability index, bins from training
    watch.py      score drift, volume, calibration as labels arrive,
                  notices as the early warning
    feeds.py      the daily price and cost check on the order feed
    forecast.py   the demand forecast's error as actuals arrive
    alerts.py     the checks, with a budget for attention
    policy.py     retraining policies, shadow scores and the gate
    report.py     the one-page report:
                  python -m foresight.monitor.report --day D
"""

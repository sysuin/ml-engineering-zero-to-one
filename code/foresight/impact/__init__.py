"""
Did it work? Foresight's impact measurement. Chapter 25 writes it.

    holdout.py    each month, hold out part of the list at random, and
                  keep the draw on record
    lists.py      v0.6's lists for past cohorts, replayed
    analyse.py    read a retention test from what Meridian records
    uplift.py     a two-model uplift sketch
    inventory.py  what ordering to a forecast costs, in dollars
    queue.py      how long an urgent ticket waits, by reading order
    report.py     the one-page impact report for finance
    simulate.py   a stand-in world for the retention test, and the
                  only module here that reads the generator's truth

No retention calls are on record at Meridian, so the retention numbers
in Chapter 25 come from a simulation, and every page this package
writes from one says so at the top.
"""

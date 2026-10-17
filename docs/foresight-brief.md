# Foresight: the renewal-risk brief

Version 0.1, written before any model. Change it in the open: date the edit and say why.

| | |
|---|---|
| Decision | Which renewals the account team calls about, before the renewal conversation |
| Who acts | The account team, working down a list delivered on Mondays |
| Prediction | The chance that a contract will not renew |
| Label | `contracts.outcome = 'not_renewed'`; contracts with no outcome yet are left out |
| Unit | One contract coming up for renewal: a row per contract, not per account |
| Moment | The 90-day mark: 90 days before `end_date`, using only what was recorded before that day |
| Windows | Observation: the year before the moment. Gap: 90 to 60 days before the end, when the team calls. Label: notice at 60 days, outcome at the end date |
| Cadence | Every Monday, score the contracts whose 90-day mark fell in the past seven days. Contracts end on month ends, so one Monday a month carries the list: about 280 contracts |
| Capacity | Ten calls a week: forty for each monthly cohort |
| Baseline | Longest gap since the last order, called first. Top 40 a cohort, renewals ending 2023 to mid-2024: precision 21.8%, against 6.5% for a random list and 45.6% for a perfect one |
| Metric | Precision among the top 40 of each cohort, pooled over cohorts, next to the baseline on the same contracts. Also reported: recall at capacity, AUC, log loss, Brier score, each with an interval |
| Costs | A call: 1.5 hours, $90. A lost renewal: $2,854 of gross profit a year (measured), saved one call in four (assumed). A wasted call costs $90; a missed leaver forgoes $624. A call pays above a 12.6% chance of leaving |
| Data | Learn from contracts ending 2023-01-01 to 2024-06-30; choose on the second half of 2024; test once on 2025 |
| Bar to clear | Beat the rule's list on the second half of 2024, on the same contracts, by more than the interval around the difference. Each point is worth about $3,400 a year |
| Out of scope | Why accounts leave and what to offer them; weighting the list by account value; demand and tickets, which get briefs of their own |
| Signs off | The head of the account team (decision, capacity), finance (costs), the engineer who owns Foresight (everything else) |

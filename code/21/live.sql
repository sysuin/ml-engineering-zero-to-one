-- The nightly scorer's order and ticket columns, written from the
-- warehouse's schema. It runs at 11 p.m. on the day of the mark, once
-- the day's orders have loaded, for the contracts that reach it.
WITH due AS (
    SELECT contract_id, account_id
    FROM contracts
    WHERE end_date = date(:mark, '+90 days')
),
placed AS (
    SELECT o.order_id, o.account_id, o.order_date,
           SUM(l.qty * l.unit_price) AS value
    FROM orders o JOIN order_lines l USING (order_id)
    WHERE o.order_date <= :mark
      AND o.account_id IN (SELECT account_id FROM due)
    GROUP BY o.order_id
)
SELECT d.contract_id,
       CAST(julianday(:mark) - julianday(MAX(p.order_date))
            AS INTEGER) AS days_since_order,
       COUNT(CASE WHEN p.order_date >= date(:mark, '-90 days')
                  THEN 1 END) AS orders_90d,
       COUNT(CASE WHEN p.order_date >= date(:mark, '-180 days')
                   AND p.order_date < date(:mark, '-90 days')
                  THEN 1 END) AS orders_prev_90d,
       ROUND(COALESCE(SUM(CASE
             WHEN p.order_date >= date(:mark, '-365 days')
             THEN p.value END), 0), 2) AS spend_365,
       (SELECT COUNT(*) FROM tickets t
         WHERE t.account_id = d.account_id
           AND date(t.opened_at) >= date(:mark, '-90 days')
           AND date(t.opened_at) <= :mark) AS tickets_90d
FROM due d LEFT JOIN placed p ON p.account_id = d.account_id
GROUP BY d.contract_id
ORDER BY d.contract_id

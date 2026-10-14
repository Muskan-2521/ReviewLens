# Hand-labelling guide (about 45 minutes for 50 reviews)

Open `data/gold_template.csv` in Excel or Google Sheets. For each row fill three columns **without looking at the model's answer**:

- `theme` - the single MAIN complaint. One of:
  `product_defect`, `performance_quality`, `delivery_logistics`, `packaging_damage`,
  `after_sales_service`, `price_value`, `usability_design`, `no_issue`, `other`
- `sentiment` - `positive`, `neutral` or `negative`
- `severity` - `0` no issue, `1` minor annoyance, `2` significant problem, `3` critical (unusable, safety risk, money lost)

Rules of thumb:
1. If a review has several complaints, pick the one the customer seems angriest about.
2. A positive review that merely mentions delivery or the battery is `no_issue`.
3. Use `other` sparingly. If you use it more than 5 times, your taxonomy may need a new theme.
4. Decide on a rule when you hit an edge case and apply it to every similar review.

Save the finished file as `data/gold_labels.csv` (CSV, same column names).

Labelling before you see the model output is what makes the accuracy number honest.

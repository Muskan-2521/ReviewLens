# Business note: what customers are really complaining about

*Fill this in after running the pipeline. One page. Write for a non-technical manager.*

## Situation
Customer reviews hold the clearest signal about product and service problems, but nobody can read thousands of them.
We tagged [N] reviews across [categories] with an LLM and ranked the issues by how often and how badly they hurt customers.

## What we found
1. **[Top issue]** - [X]% of all reviews, average severity [Y]. Concentrated in [category]. Example: "[short quote]".
2. **[Second issue]** - ...
3. **[Third issue]** - ...

## Recommendations
| Issue | Recommended action | Owner | Expected effect |
|---|---|---|---|
| [issue 1] | [specific action, e.g. add installation SLA tracking] | [team] | [e.g. fewer 1-star reviews] |
| [issue 2] | | | |
| [issue 3] | | | |

## How reliable is this?
The LLM's theme accuracy against [50] hand-labelled reviews was [X]% (keyword baseline: [Y]%).
Main confusions: [e.g. mixes up product_defect and performance_quality].
Limitations: [sample size, one-time snapshot, reviews skew negative, etc.].

## Time and cost
Manual tagging of [N] reviews would take roughly [hours] (assume [seconds] per review). The pipeline took [minutes] and cost [about X] in API usage.

## Next steps
- Expand the sample / run monthly
- Add [warranty / spare parts] as separate themes
- Pilot the top recommendation with [team]

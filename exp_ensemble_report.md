# Fold-safe ensemble audit

This audit uses `cache/ltr/standard_oof.npz` as the anchor and
`standard_svm` from `cache/supervised_multilabel_scores.npz` as the only
supervised auxiliary channel. Both are predictions from the same five IID
KFold validation partitions (`random_state=2466955`). No group-fold or
validation-label feature is used.

## Anchor

- standard LTR OOF MAP@10: **0.615924**
- standalone standard SVM OOF MAP@10: **0.606080**

The weaker SVM is still useful because its ranking errors are not identical to
the LTR errors.

## Recommended conservative rule

1. Keep exactly the ten article IDs already returned by the public baseline.
2. Convert their baseline ranks to `1 / log2(rank + 1)` scores.
3. Compute the same reciprocal-rank scores for the top 40 SVM labels.
4. Reorder the baseline ten by `baseline_score + 0.9 * svm_score`.
5. Apply the reorder only when the overlap of the two top-3 lists is not
   exactly one. When overlap is exactly one, keep the baseline order.

This rule never injects a new article, so it cannot reduce the already strong
candidate recall of the public baseline.

## Evidence

- OOF MAP@10: **0.632519**
- absolute gain: **+0.016596**
- fold gains: **+0.013982, +0.010997, +0.011101, +0.015944, +0.030954**
- query bootstrap 95% CI: **+0.006636 .. +0.027104**
- label-set cluster bootstrap 95% CI: **+0.006660 .. +0.025894**
- cluster-bootstrap probability of a positive gain: **99.94%**
- gain after removing the largest label set (`1951`, 44 rows): **+0.015480**

Thus the gain is not the earlier failure mode where almost all improvement was
caused by one repeated intent.

On OOF the rule changes top-1 for only 43/500 rows. Applied to
`best_public_061.csv`, it would keep the same ten IDs for every query, reorder
384/500 rows, and change top-1 for only 37/500 rows. This is a diagnostic only;
the audit deliberately does not write a submission.

## Alternatives rejected

- Weight 1.4 reaches 0.634163 OOF, but changes public top-1 for 120 rows and its
  label-set bootstrap lower bound nearly touches zero. The extra point estimate
  is not worth the transfer risk on the final attempt.
- Free replacement from the SVM top-40 reaches about 0.6335 OOF, but candidate
  replacement is unnecessary risk because retrieval recall is already high.
- Locking baseline top-1 drops the gain to roughly +0.005 and is negative on
  one fold. The recommended weight 0.9 is the compromise that changes top-1
  very rarely while preserving a five-fold positive result.
- Co-occurrence channels add at most small, unstable gains and are inferior to
  the pure rank-consensus rule.

The full grid, LOFO audit, bootstraps, and test change counts are stored in
`exp_ensemble_report.json`; per-row OOF audit arrays are in
`exp_ensemble_oof.npz`. `exp_ensemble_audit.py` reproduces them and never
creates `answer.csv`.

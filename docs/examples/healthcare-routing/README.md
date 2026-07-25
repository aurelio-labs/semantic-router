# Healthcare Administrative Routing

This example shows how to use Semantic Router to direct synthetic member and
provider requests to seven healthcare administrative workflows:

- `claims`
- `eligibility`
- `prior_authorization`
- `benefits`
- `pharmacy`
- `infusion_therapy`
- `appeals`

The example is administrative only. It does not diagnose conditions, recommend
treatment, make coverage decisions, or establish compliance with any privacy or
healthcare regulation.

## Contents

- `healthcare-administrative-routing.ipynb` defines the routes, runs a fully
  local encoder, optimizes route thresholds, evaluates a held-out split, explores
  ambiguous inputs, and maps route names to placeholder workflows.
- `healthcare-routing-evaluation.csv` contains synthetic member and provider
  utterances with expected route labels.

All utterances were authored for this example. They are not copied, derived, or
transformed from patient, member, provider, claim, or call-center records and
contain no protected health information (PHI).

## Route boundaries

The routes intentionally overlap. Route by the **administrative action requested**
rather than by a healthcare noun alone.

| Route | Use when the requested action is | Do not use when |
| --- | --- | --- |
| `claims` | Checking submission, adjudication, payment, adjustment, or denial of a claim after a service or item was billed | The request is about approval before service or formally disputing a completed decision |
| `eligibility` | Confirming whether a person is enrolled or eligible on a date | The person is asking what a covered plan pays or what limits apply |
| `prior_authorization` | Starting, checking, or supplying information for approval required before a service or drug | The request is about a claim already billed or a formal appeal |
| `benefits` | Explaining coverage, cost share, limits, exclusions, or network rules | The question is only whether enrollment is active |
| `pharmacy` | Handling formulary, dispensing, refill, quantity limit, or retail/mail-order pharmacy administration | The primary requested action is prior authorization, infusion scheduling, or an appeal |
| `infusion_therapy` | Coordinating an infusion site, scheduling, drug delivery to the site, or infusion-specific administration | The request is solely about general drug coverage, prior authorization, a claim, or an appeal |
| `appeals` | Challenging, reconsidering, or formally reviewing an existing adverse determination | The person only wants the reason/status of an initial claim or authorization |

### Boundary and precedence rules

1. Explicit requested action wins over the service or product mentioned.
   “Appeal the denial of my infusion” routes to `appeals`, while “schedule my
   approved infusion” routes to `infusion_therapy`.
2. `eligibility` answers **whether enrollment is active**. `benefits` answers
   **what the plan covers or pays**.
3. `prior_authorization` occurs before the requested service or dispensing.
   `claims` applies after billing or claim submission.
4. A denial is not automatically an appeal. Route to `appeals` only when the
   person asks to challenge or reconsider the decision.
5. When the requested action cannot be determined, return no route and ask a
   clarifying question. Do not force a high-stakes ambiguous request into the
   nearest route.

These rules are illustrative. A production taxonomy must reflect the
organization's contracts, operations, escalation paths, and applicable
requirements.

## Evaluation dataset

The CSV columns are:

| Column | Meaning |
| --- | --- |
| `id` | Stable synthetic example identifier |
| `split` | `train`, `validation`, or `test` |
| `speaker` | `member`, `provider`, or `other` |
| `example_type` | `positive`, `negative`, or `ambiguous` |
| `utterance` | Synthetic natural-language request |
| `expected_route` | One of the seven routes, or empty when the router should abstain |
| `alternate_routes` | Plausible competing routes separated by `\|` |
| `rationale` | Short explanation of the expected behavior |

Route seed utterances in the notebook are separate from the evaluation rows to
avoid direct example leakage. Threshold fitting uses only the training split.
Validation and test examples remain held out. The notebook fixes the random seed
used by threshold search so reruns are reproducible for the same dependency and
model versions.

The dataset is deliberately small and educational. Its results must not be
interpreted as production performance.

The checked-in notebook output from a CPU run with
`BAAI/bge-small-en-v1.5` reports 83.3% held-out test accuracy and 95.8%
coverage. The optimized validation accuracy is 54.2%, and abstention recall is
0.33, so the example still needs broader domain data and calibration before any
production use. Exact results can vary with dependency and model versions.

## Run locally

Use Python 3.12 or earlier because the repository's local model dependencies are
currently restricted to Python versions below 3.13. From the repository root:

```bash
uv sync --extra local
uv run jupyter notebook \
  docs/examples/healthcare-routing/healthcare-administrative-routing.ipynb
```

The example uses `LocalEncoder` with `BAAI/bge-small-en-v1.5`. Inference runs
locally after the model is available, but the first run may download model
artifacts. Downloading a model and running it locally does not by itself
establish HIPAA compliance or authorize processing PHI.

## Extending the taxonomy

1. Begin with the operational destination and the action it owns.
2. Write a positive boundary and at least one explicit exclusion.
3. Add seed utterances from both member and provider perspectives.
4. Add boundary pairs that differ only in the requested action.
5. Add out-of-scope examples expected to return no route.
6. Preserve independent training, validation, and test splits.
7. Refit thresholds after changing routes, examples, or the encoder.
8. Review per-route precision and recall, abstention behavior, coverage, and the
   confusion matrix—not accuracy alone.
9. Version the taxonomy and dataset together.
10. Require human review before connecting route output to consequential
    healthcare actions.

Possible extensions include coordination of benefits, provider enrollment,
referrals, grievances, billing questions, durable medical equipment, dental,
vision, and behavioral-health administration. Add them only when their
ownership and boundaries can be stated clearly.

## Limitations

- The example performs single-turn semantic routing and does not use conversation
  history, authenticated identity, plan state, claim state, or authorization
  state.
- Similarity scores are not calibrated probabilities.
- Threshold optimization can overfit a small or unrepresentative dataset.
- A single-label route cannot fully represent multi-intent requests.
- Local inference does not guarantee privacy, security, or regulatory
  compliance.
- The example does not identify emergencies, provide clinical advice, determine
  medical necessity, authorize care, adjudicate claims, or decide appeals.
- Production systems need monitoring, audit logging, access controls, data
  governance, drift testing, fallback behavior, and human escalation.

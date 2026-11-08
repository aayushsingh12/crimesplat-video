# ADR-0005 — Human review of all AI-proposed content

- **Status:** accepted
- **Date:** 2026-11-07

## Context
Open-vocabulary detectors and VLMs produce false positives and hallucinations (R-22). In a forensic setting, a fabricated "knife" pin is worse than a missing one.

## Decision
Every automatically generated evidence point, event and actor track is created with `status: "proposed"`. The published viewer shows only `confirmed` items by default. Proposed items, when shown, are visually distinct and labelled "AI". All review actions are logged with reviewer and timestamp.

## Alternatives considered
- Fully automatic publishing with a confidence threshold: not acceptable for the use case.
- Manual-only annotation: loses the research contribution and the time savings.

## Consequences
- ✅ Trustworthy output. Measurable reviewer time savings (RQ3).
- ❌ Requires building the review tool. Throughput is limited by the reviewer.

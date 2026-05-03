# RFC Process — PCL Rustic

This directory contains **Request for Comments (RFC)** documents for PCL Rustic. RFCs are the primary mechanism for proposing, discussing, and deciding on significant design and architecture changes.

---

## RFC Instructions (always keep this section up to date)

### When to write an RFC

Write an RFC when you want to make a **significant** change that affects:

- Public API surface (new types, methods, behaviours, or breaking changes)
- Core architecture or Rust trait hierarchy
- Cross-cutting concerns (error handling strategy, memory layout, tensor backend)
- CI/CD or tooling decisions that affect all contributors
- Performance characteristics or guarantees

Small bug fixes, documentation updates, and refactors that preserve existing behaviour **do not** need an RFC.

### RFC numbering

RFCs are numbered sequentially from `RFC-0001`. Use zero-padded 4-digit numbers. When drafting, pick the next available number.

### RFC lifecycle

```
Draft → Open for Review → Accepted | Rejected | Withdrawn
```

| Status | Meaning |
|--------|---------|
| `Draft` | Author is still working on the document |
| `Open for Review` | Ready for community feedback. **Requires 2 independent sub-agent reviews.** |
| `Accepted` | Approved; implementation may begin |
| `Rejected` | Decided not to proceed |
| `Withdrawn` | Author pulled back the proposal |
| `Implemented` | RFC is fully implemented in the codebase |

### Review process

Every RFC that reaches `Open for Review` **must** receive reviews from **2 independent sub-agents** (or human reviewers). Each reviewer must:

1. Confirm the motivation is clear and valid.
2. Evaluate the design for correctness, feasibility, and impact on existing code.
3. Check for unresolved questions that must be answered before acceptance.
4. Leave explicit `APPROVED` or `CHANGES REQUESTED` along with comments.

Only after both reviewers give `APPROVED` may the RFC be moved to `Accepted`.

### File naming convention

```
rfcs/RFC-NNNN-short-title.md
```

Examples:
- `rfcs/RFC-0001-core-architecture.md`
- `rfcs/RFC-0002-gpu-acceleration.md`

### Bidirectional links

If an RFC is related to another RFC, **both** documents must contain a **Related RFCs** section that links to each other. Links use relative paths:

```markdown
## Related RFCs

- [RFC-0001 — Core Architecture](./RFC-0001-core-architecture.md) — provides the trait foundations used here
- [RFC-0003 — Advanced Downsampling](./RFC-0003-advanced-downsampling.md) — depends on the GPU backend introduced in this RFC
```

### Tooling conventions

- **API documentation**: use [context7](https://context7.com) to look up the most up-to-date docs for Rust crates (Burn, PyO3, ndarray, etc.) before finalising design choices.
- **Integrated tests**: all new functionality described in an RFC must have integration tests written in **pytest** and wired into CI.
- **Batch shell operations**: use [`justfile`](../justfile) recipes instead of plain shell scripts or Makefiles.

### RFC template

```markdown
# RFC-NNNN — Title

| Field | Value |
|-------|-------|
| RFC | NNNN |
| Status | Draft |
| Author | <name> |
| Created | YYYY-MM-DD |
| Updated | YYYY-MM-DD |

## Summary

One paragraph describing the proposal.

## Motivation

Why is this change needed? What problem does it solve?

## Design

Detailed technical design. Include Rust type signatures, Python API examples, and
trait changes where relevant.

## Drawbacks

What are the costs and risks?

## Alternatives

What other approaches were considered and why were they rejected?

## Unresolved Questions

List open questions that need to be settled before acceptance.

## Related RFCs

Links to connected RFCs (bidirectional).

## Reviews

Record of sub-agent reviews (add after submitting for review).
```

---

## RFC Index

| RFC | Title | Status |
|-----|-------|--------|
| [RFC-0001](./RFC-0001-core-architecture.md) | Core Architecture & Design Decisions | Implemented |
| [RFC-0002](./RFC-0002-gpu-acceleration.md) | GPU Acceleration Backend | Open for Review |
| [RFC-0003](./RFC-0003-advanced-downsampling.md) | Advanced Downsampling Strategies (FPS & Normal-based) | Open for Review |
| [RFC-0004](./RFC-0004-point-cloud-registration.md) | Point Cloud Registration Algorithms (ICP & NDT) | Open for Review |
| [RFC-0005](./RFC-0005-normal-estimation.md) | Normal Vector Estimation | Open for Review |
| [RFC-0006](./RFC-0006-point-cloud-segmentation.md) | Point Cloud Segmentation | Open for Review |
| [RFC-0007](./RFC-0007-testing-ci-infrastructure.md) | Testing & CI Infrastructure Enhancement | Open for Review |

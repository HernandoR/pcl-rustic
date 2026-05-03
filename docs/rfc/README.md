# PCL Rustic RFCs (Request for Comments)

This directory contains Request for Comments (RFC) documents for major design decisions in the PCL Rustic project. Each RFC records the motivation, design, trade-offs, and final decision for a significant architectural or feature decision.

---

## RFC Process Instructions

### What Deserves an RFC?

An RFC is required for any change that:
- Introduces a new public API or alters an existing one
- Modifies the Rust–Python interop layer (PyO3 bindings, type stubs)
- Adds a new algorithmic subsystem (downsampling strategy, I/O format, registration, etc.)
- Changes the build, test, or CI infrastructure in a non-trivial way
- Introduces a new external dependency (Rust crate or Python package)
- Affects performance characteristics or memory layout guarantees

Small bug-fixes, documentation updates, and refactors that do not change observable behaviour do **not** need an RFC.

### Lifecycle

```
Draft → Review → Accepted / Rejected / Withdrawn → Implemented
```

| Status | Meaning |
|--------|---------|
| `Draft` | Initial proposal, open for discussion |
| `Review` | Assigned reviewers are providing feedback |
| `Accepted` | Approved by maintainers; implementation may begin |
| `Rejected` | Proposal was declined (with reasoning) |
| `Withdrawn` | Author withdrew the proposal |
| `Implemented` | Feature is shipped and RFC is closed |
| `Superseded` | Replaced by a newer RFC (link provided) |

### Numbering

RFCs are numbered sequentially: `RFC-XXXX` where `XXXX` is a zero-padded four-digit number (e.g., `RFC-0001`). The next available number is **RFC-0008**.

### File Naming

```
docs/rfc/RFC-XXXX-short-title.md
```

### Template

Copy the following template to start a new RFC:

```markdown
# RFC-XXXX: Title

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author(s)** | Your Name |
| **Created** | YYYY-MM-DD |
| **Updated** | YYYY-MM-DD |
| **Related RFCs** | [RFC-XXXX](RFC-XXXX-title.md) |

## Summary

One-paragraph executive summary.

## Motivation

Why is this change needed? What problems does it solve?

## Design

### Overview

High-level architecture.

### Detailed Design

Rust API, Python bindings, file structures, etc.

### Alternatives Considered

What other approaches were evaluated and why they were not chosen.

## Impact

### Breaking Changes

List any breaking changes to existing public APIs.

### Performance Implications

Expected impact on throughput, latency, or memory usage.

### Testing Plan

How will this be tested? (unit tests, integration pytest, CI, benchmarks)

## Unresolved Questions

Open questions that must be answered before implementation.

## References

Links to relevant papers, crates, or prior art.
```

### Review Requirements

Every new RFC (**Draft** state) must be reviewed by **at least 2 independent reviewers** before it can move to **Accepted**. Reviews are recorded inline in the RFC under a `## Review Notes` section, or as PR comments.

### Tooling Notes

- **API Documentation**: Use [context7](https://context7.io) to retrieve up-to-date API docs for external crates and Python packages referenced in RFCs.
- **Integration Tests**: All accepted RFCs that involve code changes must include a pytest integration test plan. Tests live in `tests/` and follow the naming convention `test_<module>.py`.
- **CI**: The `justfile` is the single source of truth for batch shell operations in CI and local development. Avoid raw `Makefile` targets or bare shell scripts.
- **Batch Operations**: Use `just <recipe>` for all build, test, format, lint, and release automation.

---

## RFC Index

| Number | Title | Status | Author |
|--------|-------|--------|--------|
| [RFC-0001](RFC-0001-core-architecture.md) | Core Architecture | Implemented | liuzhen19 |
| [RFC-0002](RFC-0002-enhanced-io.md) | Enhanced I/O Subsystem | Draft | liuzhen19 |
| [RFC-0003](RFC-0003-advanced-downsampling.md) | Advanced Downsampling Strategies | Draft | liuzhen19 |
| [RFC-0004](RFC-0004-point-cloud-registration.md) | Point Cloud Registration | Draft | liuzhen19 |
| [RFC-0005](RFC-0005-normal-estimation.md) | Normal Vector Estimation | Draft | liuzhen19 |
| [RFC-0006](RFC-0006-gpu-acceleration.md) | GPU Acceleration Backend | Draft | liuzhen19 |
| [RFC-0007](RFC-0007-testing-infrastructure.md) | Testing & Developer Infrastructure | Draft | liuzhen19 |

---

*Last updated: 2026-05-03*

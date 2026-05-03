# PCL Rustic RFCs (Request for Comments)

This directory contains the RFC documents for the PCL Rustic project. RFCs describe significant design decisions, new features, and architectural changes to the library.

## RFC Process

### What is an RFC?

An RFC (Request for Comments) is a document that proposes a significant change or addition to PCL Rustic. RFCs are used to:

- Propose new features or algorithms
- Describe architectural changes
- Document design decisions and their rationale
- Gather community feedback before implementation

### RFC Lifecycle

```
Draft → Proposed → Accepted → Implemented
                ↘ Rejected
                ↘ Deferred
```

| Status        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| **Draft**     | Initial writing; author is still developing the idea                        |
| **Proposed**  | Ready for community review; open for comments and feedback                  |
| **Accepted**  | Approved for implementation                                                 |
| **Rejected**  | Not accepted; reason documented in the RFC                                  |
| **Deferred**  | Valid idea but not prioritized; may be revisited                            |
| **Implemented** | Fully implemented in codebase; linked to relevant commits/releases       |
| **Superseded** | Replaced by a newer RFC; link provided to successor                       |
| **Deprecated** | Feature/design removed; reason documented                                 |

### Numbering

RFCs are numbered sequentially: `RFC-NNNN-short-title.md` (e.g., `RFC-0001-initial-architecture.md`).

### RFC Template

Every RFC must follow this template (see below). Copy it when drafting a new RFC.

```markdown
# RFC-NNNN: Title

| Field     | Value                             |
|-----------|-----------------------------------|
| Status    | Draft                             |
| Authors   | Name <email>                      |
| Created   | YYYY-MM-DD                        |
| Updated   | YYYY-MM-DD                        |

## Summary

One-paragraph summary of the proposal.

## Motivation

Why is this change needed? What problem does it solve?

## Design Details

Detailed description of the proposed change.

## Alternatives Considered

What other approaches were evaluated, and why were they rejected?

## Open Questions

Questions that need to be resolved before implementation.

## Implementation Plan

- [ ] Step 1
- [ ] Step 2

## References

- Related RFC: [RFC-NNNN](./RFC-NNNN-title.md)
- External links
```

### Review Process

1. Author creates a new branch and drafts the RFC.
2. Author opens a PR targeting `main`; 2 independent reviewers are assigned.
3. Reviewers leave comments; author addresses them.
4. Once both reviewers approve, the RFC status moves to **Accepted**.
5. Implementation can begin. When done, status changes to **Implemented**.

### Bidirectional Links

When an RFC references or depends on another RFC, **both** documents must link to each other:

```markdown
## Related RFCs

- **Depends on**: [RFC-0001](./RFC-0001-initial-architecture.md)
- **Informs**: [RFC-0004](./RFC-0004-point-cloud-registration.md)
```

### Tooling Conventions

All tooling decisions in RFCs follow these project-wide conventions:

- **Build tool**: [`justfile`](../justfile) (never Makefile)
- **Testing**: `pytest` with `uv run pytest` — integrated into CI
- **API reference**: Use [context7](https://context7.com) MCP server for up-to-date library API docs
- **Python package manager**: `uv`
- **Rust**: `cargo` + `maturin` for PyO3 extension build

### CI Integration

Tests specified in an RFC's implementation plan must be added as `pytest` tests and must pass in the GitHub Actions CI pipeline (`.github/workflows/test.yml`).

---

## Index

| RFC | Title | Status |
|-----|-------|--------|
| [RFC-0001](./RFC-0001-initial-architecture.md) | Initial Architecture and Design Decisions | Implemented |
| [RFC-0002](./RFC-0002-gpu-acceleration.md) | GPU Acceleration Support | Draft |
| [RFC-0003](./RFC-0003-downsampling-strategies.md) | Additional Downsampling Strategies | Draft |
| [RFC-0004](./RFC-0004-point-cloud-registration.md) | Point Cloud Registration (ICP / NDT) | Draft |
| [RFC-0005](./RFC-0005-normal-estimation.md) | Normal Vector Estimation | Draft |
| [RFC-0006](./RFC-0006-segmentation.md) | Point Cloud Segmentation | Draft |
| [RFC-0007](./RFC-0007-parquet-format.md) | Parquet Format I/O | Draft |

# PCL Rustic — RFC Process

This directory contains **Request for Comments (RFC)** documents for the PCL Rustic project.  
Every significant design decision, new feature area, or architectural change should be captured in an RFC before implementation begins.

---

## 📋 RFC Instructions (Always Read Before Drafting)

### What Goes in an RFC?

An RFC describes **why** a change is needed and **how** it will be designed — not line-by-line code.  
It is the _single source of truth_ for major design decisions.

An RFC is required when:
- Adding a new top-level feature area (new module, new algorithm family)
- Changing a public API in a breaking or non-trivial way
- Adopting a new external dependency or technology (e.g., GPU backend)
- Changing the build system, CI pipeline, or release process significantly

An RFC is _not_ required for:
- Bug fixes
- Documentation-only changes
- Trivial refactors with no API impact

### RFC Status Lifecycle

```
Draft → Under Review → Accepted → Implemented
                    → Rejected
                    → Superseded (by RFC-XXXX)
```

### RFC Template

Every RFC must follow the structure in [RFC-TEMPLATE.md](./RFC-TEMPLATE.md).

### Naming Convention

`RFC-XXXX-short-title.md` where XXXX is a zero-padded four-digit number.

### Bidirectional Links

If RFC-B is related to or depends on RFC-A:
- RFC-A must have a **"Related RFCs"** section listing RFC-B
- RFC-B must have a **"Related RFCs"** section listing RFC-A

### Review Process

1. Author drafts the RFC and opens a PR.
2. **Two separate sub-agents** (or human reviewers) each provide independent review.
3. Author incorporates feedback; unresolved questions must be noted.
4. RFC status moves to **Accepted** when consensus is reached.
5. Implementation PRs must reference the accepted RFC number.

---

## 🛠 Tooling Conventions

| Tool | Purpose | Command |
|------|---------|---------|
| `just` (justfile) | All batch shell operations (build, test, lint, release) | `just --list` |
| `pytest` | Integrated Python tests including CI | `just test` |
| `context7` MCP | Up-to-date API docs lookup | Use in Copilot agent sessions |
| `ruff` | Python linting & formatting | `just lint` |
| `cargo clippy` | Rust linting | `just lint` |
| `maturin` | Rust→Python wheel build | `just build` |

> **Important for agents:** Always use `justfile` for batch operations, never `Makefile`.  
> Always use `pytest` (via `just test`) for integration tests, never ad-hoc shell scripts.  
> When looking up PyO3, Burn, or NumPy API details, prefer the `context7` MCP tool for the most up-to-date docs.

---

## 📚 RFC Index

| RFC | Title | Status | Related |
|-----|-------|--------|---------|
| [RFC-0001](./RFC-0001-foundation.md) | Project Foundation & Architecture | Accepted | RFC-0002, RFC-0003 |
| [RFC-0002](./RFC-0002-tensor-backend.md) | Tensor Backend & GPU Acceleration | Draft | RFC-0001, RFC-0004, RFC-0005, RFC-0006, RFC-0007 |
| [RFC-0003](./RFC-0003-extended-io.md) | Extended File Format Support | Draft | RFC-0001 |
| [RFC-0004](./RFC-0004-sampling-strategies.md) | Advanced Downsampling Strategies | Draft | RFC-0001, RFC-0002 |
| [RFC-0005](./RFC-0005-registration.md) | Point Cloud Registration (ICP, NDT) | Draft | RFC-0001, RFC-0002, RFC-0006 |
| [RFC-0006](./RFC-0006-normal-estimation.md) | Normal Vector Estimation | Draft | RFC-0001, RFC-0002, RFC-0005 |
| [RFC-0007](./RFC-0007-segmentation.md) | Point Cloud Segmentation & Clustering | Draft | RFC-0001, RFC-0002, RFC-0006 |

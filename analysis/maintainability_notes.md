# Maintainability Experiment Notes

## Overview

This document records the results of three representative modification
experiments (M1, M2, M3) performed on both the Monolithic baseline and
the PatternRAG architecture.

These experiments are the primary evidence for the research claim that
a pattern-based architecture is more maintainable and extensible.

---

## Experiment Protocol

For each task:
1. Start from a clean git state on the stable implementation branch.
2. Perform the modification independently on the monolithic system.
3. Commit, record metrics, reset to clean state.
4. Perform the modification independently on the PatternRAG system.
5. Commit, record metrics, reset to clean state.

Metrics to record (via `git diff --stat` before committing):
- Files changed (count)
- Lines added
- Lines deleted
- Existing classes/functions modified (Yes/No, count)
- New files created (count)
- New classes/functions created (count)
- Approximate implementation time (minutes)

---

## M1 — Add a New Retrieval Strategy

**Task**: Add a TF-IDF retriever as an alternative to BM25.

### Monolithic Results

| Metric | Value |
|---|---|
| Files changed | |
| Lines added | |
| Lines deleted | |
| Existing classes modified | |
| New files created | |
| New classes created | |
| Implementation time (min) | |
| Notes | |

### PatternRAG Results

| Metric | Value |
|---|---|
| Files changed | |
| Lines added | |
| Lines deleted | |
| Existing classes modified | |
| New files created | |
| New classes created | |
| Implementation time (min) | |
| Notes | |

### M1 Analysis

*(Fill in after performing the experiment)*

---

## M2 — Add a New Cross-cutting Concern

**Task**: Add a query-rewriting step (e.g., expand abbreviations or
rephrase for better retrieval) as a decorator/wrapper around retrieval.

### Monolithic Results

| Metric | Value |
|---|---|
| Files changed | |
| Lines added | |
| Lines deleted | |
| Existing classes modified | |
| New files created | |
| New classes created | |
| Implementation time (min) | |
| Notes | |

### PatternRAG Results

| Metric | Value |
|---|---|
| Files changed | |
| Lines added | |
| Lines deleted | |
| Existing classes modified | |
| New files created | |
| New classes created | |
| Implementation time (min) | |
| Notes | |

### M2 Analysis

*(Fill in after performing the experiment)*

---

## M3 — Add a New Monitoring Component

**Task**: Add a hit-rate counter that tracks what fraction of retrieved
documents are relevant (based on gold IDs).

### Monolithic Results

| Metric | Value |
|---|---|
| Files changed | |
| Lines added | |
| Lines deleted | |
| Existing classes modified | |
| New files created | |
| New classes created | |
| Implementation time (min) | |
| Notes | |

### PatternRAG Results

| Metric | Value |
|---|---|
| Files changed | |
| Lines added | |
| Lines deleted | |
| Existing classes modified | |
| New files created | |
| New classes created | |
| Implementation time (min) | |
| Notes | |

### M3 Analysis

*(Fill in after performing the experiment)*

---

## Summary Table

*(To be filled in after all three experiments are complete)*

| Experiment | System | Files Changed | LOC Added | Existing Code Modified | New Components |
|---|---|---|---|---|---|
| M1 | Monolithic | | | | |
| M1 | PatternRAG | | | | |
| M2 | Monolithic | | | | |
| M2 | PatternRAG | | | | |
| M3 | Monolithic | | | | |
| M3 | PatternRAG | | | | |

---

## Extensibility Notes

*(Fill in qualitative observations about how each system responds to
change — e.g., whether adding a feature required modifying existing code,
whether components could be tested in isolation, etc.)*

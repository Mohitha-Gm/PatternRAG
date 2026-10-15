# PatternRAG Evaluation Dashboard (UI)

A dedicated, lightweight, and modern UI dashboard built to showcase all experimental results, retrieval metrics, RAGAS quality scores, latency breakdowns, and architectural extensibility comparisons for **PatternRAG**.

---

## How to View the Dashboard

This UI is **100% self-contained** and has **zero build steps or package installations required**.

### Option 1: Direct Double-Click (No server needed!)
Simply open `index.html` in your browser:
* Double-click [index.html](file:///c:/Users/mohit/OneDrive/Desktop/INCOMPLETE%20PROJ/PatternRAG/ui/index.html) in your Windows File Explorer, **OR**
* Right-click `index.html` → *Open with* → *Google Chrome* / *Microsoft Edge* / *Firefox*.

### Option 2: Local HTTP Server (Optional)
If you prefer running via a quick local web server:
```bash
# In your terminal from the project root:
cd ui
python -m http.server 8000
```
Then visit: `http://localhost:8000`

---

## Dashboard Highlights & Features

1. **Executive Summary & Key Findings**:
   * Instant view of the 100% Hybrid Retrieval Equivalence guarantee.
   * MRR, NDCG@10, and RAGAS scores summary cards.
   * Key takeaways addressing the primary research question.

2. **Retrieval Benchmark**:
   * Side-by-side comparison table of all IR metrics (MRR, NDCG@10, Precision@1/5/10, Recall@1/5/10).
   * Visual percentage meters and custom interactive SVG grouped bar charts.

3. **RAGAS Answer-Quality Evaluation**:
   * Evaluation breakdown using `openai/gpt-oss-120b` across Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

4. **Latency & Cost Profile**:
   * Detailed breakdown across Retrieval, Generation, and Total pipeline runtime (Mean, Median, P95).
   * Decoupled telemetry cards showing token counts and estimated API costs from `CostMonitor`, `LatencyLogger`, and `MetricsCollector` observers.

5. **30-Query Deep Dive Explorer**:
   * Search through all 30 evaluated queries by question keyword or answer text.
   * Filter queries by agreement, disagreement, or high-faithfulness scores.
   * Side-by-side comparison of **Monolithic Predicted Answer** vs **PatternRAG Predicted Answer** against the Ground Truth.
   * Clickable toggle to view the top-10 retrieved document IDs for every query, demonstrating exact rank equivalence.

6. **Architectural Extensibility (M1–M3)**:
   * Interactive cards explaining the 4 Gang of Four (GoF) design patterns: **Strategy**, **Decorator**, **Observer**, and **Factory**.
   * Concrete comparison of the 3 modification experiments demonstrating change localization: **0 lines changed in core PatternRAG pipeline** vs invasive changes in Monolithic.

7. **Figures & Visualizations**:
   * High-resolution embedded views of `figures/retrieval_metrics.png` and `figures/latency_comparison.png` with a click-to-enlarge lightbox modal.

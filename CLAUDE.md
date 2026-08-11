# Engineering Principles

## Code Style
- Write like a principal engineer: readable and brief over verbose or clever
- Prefer simple, obvious solutions — add complexity only when the user approves
- Don't defensively code for every edge case upfront; handle the happy path first

### Good vs Bad Examples

**Readability over cleverness**
```python
# Bad — clever but unreadable
result = {k: v for d in dicts for k, v in d.items() if v is not None and k not in seen and not seen.add(k)}

# Good — obvious at a glance
merged = {}
for d in dicts:
    for k, v in d.items():
        if v is not None and k not in merged:
            merged[k] = v
```

**Simple first, complexity on request**
```python
# Start here — works for the common case
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# Add only when the user says "we need chunked loading for large files"
def load_csv(path: str, chunksize: int | None = None) -> pd.DataFrame | Iterator:
    ...
```

**Don't over-engineer edge cases**
```python
# Bad — defensively handles every edge case nobody asked for
def normalize_column(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[col].isna().all():
        return pd.Series(dtype=float)
    ...

# Good — handle the real case; let errors surface naturally
def normalize_column(df: pd.DataFrame, col: str) -> pd.Series:
    values = df[col].astype(float)
    return (values - values.mean()) / values.std()
```

---

## Testing
- Write meaningful tests that target genuinely complex logic
- Optimize for testing hard-to-reason-about behavior, not coverage metrics
- Avoid trivial tests that just restate what the code already makes obvious

### Good vs Bad Examples

**Test complex logic, not trivial wrappers**
```python
# Bad — tests nothing beyond the pandas API itself
def test_load_csv():
    df = load_csv("data.csv")
    assert isinstance(df, pd.DataFrame)

# Good — tests the tricky normalization edge case (zero std → NaN vs error)
def test_normalize_constant_column():
    df = pd.DataFrame({"x": [5, 5, 5]})
    result = normalize_column(df, "x")
    assert result.isna().all()  # std=0, behavior is ambiguous — pin it
```

**Test data transformations with representative fixtures**
```python
# Good — uses a realistic fixture that covers the join's nullable key behavior
def test_enrich_events():
    events = pd.DataFrame({"user_id": [1, 2, None], "value": [10, 20, 30]})
    users  = pd.DataFrame({"user_id": [1, 2],       "tier": ["A", "B"]})
    result = enrich_events(events, users)
    assert result["tier"].isna().sum() == 1   # unmatched row stays in output
    assert len(result) == len(events)          # no rows dropped
```

**Pin parsing behavior that could silently regress**
```python
# Good — LLM/parser output format is fragile; pin the exact contract
def test_parse_chart_response_handles_markdown_fence():
    raw = '```json\n{"labels": ["A"], "values": [1]}\n```'
    result = parse_chart_response(raw)
    assert result == ChartData(labels=["A"], values=[1])
```

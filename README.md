# Portfolio demand scenarios

A small decision-support app: given a portfolio of client molecules at different development phases,
how much manufacturing capacity will be needed each year, and how likely is a shortfall?

- **Model** (`model.py`): each molecule succeeds with the probability of reaching approval from its
  phase; if it succeeds, demand ramps linearly from launch to peak. A Monte Carlo draws which molecules
  succeed and gives P10 / P50 / P90 demand per year and P(demand > capacity).
- **Scenarios** (sidebar): success-rate multiplier, launch delay, peak demand vs plan, capacity.
- **App** (`app.py`): Streamlit + Plotly; portfolio and success rates are editable tables; results download as CSV.
- **Tests** (`tests/`): the Monte Carlo mean matches the analytic expectation, delays shift demand,
  probabilities are capped, bad input is rejected with a clear message.
- **CI** (`.github/workflows/ci.yml`): runs the tests and builds the Docker image on every push.

The portfolio in `data/portfolio.csv` is synthetic and the default success rates are illustrative;
replace both with real data and your own benchmarks before using the numbers.

## Run

```bash
pip install -r requirements.txt pytest
pytest -q
streamlit run app.py
```

or with Docker:

```bash
docker build -t portfolio-demand-demo .
docker run -p 8501:8501 portfolio-demand-demo
```

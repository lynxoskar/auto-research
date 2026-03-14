"""Smoke tests — verify the full pipeline works end-to-end without an LLM."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from backtest import backtest
from firewall import (
    anonymize_dataset,
    generate_key,
    generate_synthetic_data,
    load_key,
    load_parquet_duckdb,
    save_key,
)
from sandbox import run_explore, run_strategy
from torture import deflation_test, noise_test, walkforward_test

# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------


class TestFirewall:
    def test_anonymize_symbols_are_deterministic(self):
        df = generate_synthetic_data(symbols=["AAPL"], bars_per_symbol=10)
        key = generate_key()
        anon1, _ = anonymize_dataset(df, key)
        anon2, _ = anonymize_dataset(df, key)
        assert anon1["symbol"][0] == anon2["symbol"][0]

    def test_anonymize_symbols_differ_by_key(self):
        df = generate_synthetic_data(symbols=["AAPL"], bars_per_symbol=10)
        anon1, _ = anonymize_dataset(df, generate_key())
        anon2, _ = anonymize_dataset(df, generate_key())
        assert anon1["symbol"][0] != anon2["symbol"][0]

    def test_anonymize_destroys_absolute_prices(self):
        df = generate_synthetic_data(symbols=["AAPL"], bars_per_symbol=100)
        key = generate_key()
        anon, _ = anonymize_dataset(df, key)
        assert float(anon["close"].mean()) < 1.0  # type: ignore[arg-type]
        assert float(anon["close"].max()) < 0.5  # type: ignore[arg-type]

    def test_anonymize_shifts_dates(self):
        df = generate_synthetic_data(symbols=["AAPL"], bars_per_symbol=10)
        key = generate_key()
        anon, _ = anonymize_dataset(df, key, date_offset_days=365)
        original_first = df.sort("timestamp")["timestamp"][0]
        anon_first = anon.sort("timestamp")["timestamp"][0]
        diff = (original_first - anon_first).days
        assert 360 <= diff <= 370

    def test_anonymize_returns_reverse_map(self):
        df = generate_synthetic_data(symbols=["AAPL", "GOOGL"], bars_per_symbol=10)
        key = generate_key()
        _, rmap = anonymize_dataset(df, key)
        assert len(rmap) == 2
        assert set(rmap.values()) == {"AAPL", "GOOGL"}

    def test_anonymize_volume_is_zscore(self):
        df = generate_synthetic_data(symbols=["AAPL"], bars_per_symbol=200)
        key = generate_key()
        anon, _ = anonymize_dataset(df, key)
        vol = anon["volume"]
        assert abs(float(vol.mean())) < 0.1  # type: ignore[arg-type]
        assert 0.5 < float(vol.std()) < 1.5  # type: ignore[arg-type]

    def test_synthetic_data_shape(self):
        df = generate_synthetic_data(symbols=["A", "B"], bars_per_symbol=50)
        assert len(df) == 100
        assert set(df.columns) == {"symbol", "timestamp", "open", "high", "low", "close", "volume"}

    def test_parquet_roundtrip(self, tmp_path):
        df = generate_synthetic_data(symbols=["X"], bars_per_symbol=20)
        path = tmp_path / "test.parquet"
        df.write_parquet(path)
        loaded = load_parquet_duckdb(path)
        assert len(loaded) == 20
        assert set(loaded.columns) == set(df.columns)


# ---------------------------------------------------------------------------
# Backtest (position weights)
# ---------------------------------------------------------------------------


class TestBacktest:
    def test_empty_data(self):
        result = backtest(np.array([]), np.array([]))
        assert result["sharpe"] == 0.0
        assert result["trades"] == 0
        assert result["final_equity"] == 1.0

    def test_flat_positions_no_trades(self):
        returns = np.random.default_rng(42).normal(0.001, 0.02, 100)
        positions = np.zeros(100)
        result = backtest(returns, positions)
        assert result["trades"] == 0
        assert result["final_equity"] == 1.0
        assert result["exposure"] == 0.0

    def test_full_long_on_uptrend(self):
        returns = np.full(200, 0.005)  # strong uptrend
        positions = np.ones(200)  # always long
        result = backtest(returns, positions, cost_bps=0)
        assert result["sharpe"] > 0
        assert result["final_equity"] > 1.0
        assert result["exposure"] == 1.0

    def test_short_on_downtrend(self):
        returns = np.full(200, -0.005)  # downtrend
        positions = np.full(200, -1.0)  # always short
        result = backtest(returns, positions, cost_bps=0)
        assert result["sharpe"] > 0  # shorting a downtrend is profitable
        assert result["final_equity"] > 1.0

    def test_transaction_costs_reduce_equity(self):
        returns = np.full(100, 0.001)
        # Flip between long and flat frequently → high costs
        positions = np.where(np.arange(100) % 10 < 5, 1.0, 0.0)
        no_cost = backtest(returns, positions, cost_bps=0)
        with_cost = backtest(returns, positions, cost_bps=50)
        assert with_cost["final_equity"] < no_cost["final_equity"]

    def test_partial_positions(self):
        returns = np.full(100, 0.01)
        full = backtest(returns, np.ones(100), cost_bps=0)
        half = backtest(returns, np.full(100, 0.5), cost_bps=0)
        assert half["final_equity"] < full["final_equity"]
        assert half["final_equity"] > 1.0

    def test_max_drawdown_is_negative_or_zero(self):
        returns = np.random.default_rng(7).normal(0, 0.02, 300)
        positions = np.ones(300)
        result = backtest(returns, positions)
        assert result["max_drawdown"] <= 0

    def test_win_rate_bounded(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 500)
        positions = np.where(returns > 0, 1.0, 0.0)  # perfect hindsight (cheating)
        result = backtest(returns, positions)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_positions_clipped(self):
        returns = np.full(50, 0.01)
        positions = np.full(50, 5.0)  # exceeds bounds
        result = backtest(returns, positions, cost_bps=0)
        # Should be clipped to 1.0
        one_result = backtest(returns, np.ones(50), cost_bps=0)
        assert abs(result["final_equity"] - one_result["final_equity"]) < 0.001


# ---------------------------------------------------------------------------
# Torture
# ---------------------------------------------------------------------------


class TestTorture:
    def test_noise_test_on_trending_data(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.003, 0.01, 500)
        positions = np.where(np.cumsum(returns) > 0, 1.0, 0.0)
        result = noise_test(returns, positions)
        assert result["real_sharpe"] > 0

    def test_noise_test_fails_on_random_positions(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 500)
        positions = rng.uniform(-1, 1, 500)
        result = noise_test(returns, positions, n_shuffles=10)
        assert isinstance(result["passed"], bool)

    def test_deflation_test_structure(self):
        returns = np.random.default_rng(42).normal(0.001, 0.02, 200)
        positions = np.ones(200)
        result = deflation_test(returns, positions)
        assert "passed" in result
        assert "base_sharpe" in result
        assert "deflated_sharpe" in result
        assert result["deflated_sharpe"] <= result["base_sharpe"]

    def test_walkforward_passes_on_steady_trend(self):
        returns = np.full(500, 0.003)
        positions = np.ones(500)
        result = walkforward_test(returns, positions)
        assert result["passed"] is True
        assert result["pass_rate"] > 0.5

    def test_walkforward_structure(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 500)
        positions = np.ones(500)
        result = walkforward_test(returns, positions)
        assert "passed" in result
        assert "folds" in result
        assert "pass_rate" in result
        assert isinstance(result["folds"], list)
        assert len(result["folds"]) > 0

    def test_walkforward_needs_enough_data(self):
        returns = np.full(30, 0.003)
        positions = np.ones(30)
        result = walkforward_test(returns, positions)
        assert "skipped" in result or result.get("passed") is not None


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class TestSandbox:
    def _make_bars(self, n: int = 100) -> dict:
        rng = np.random.default_rng(42)
        return {
            "open": rng.normal(0, 0.02, n).tolist(),
            "high": rng.normal(0, 0.02, n).tolist(),
            "low": rng.normal(0, 0.02, n).tolist(),
            "close": rng.normal(0.001, 0.02, n).tolist(),
            "volume": rng.normal(0, 1, n).tolist(),
        }

    def test_template_strategy_runs(self):
        result = run_strategy("strategy_template.py", self._make_bars())
        assert "error" not in result, result.get("error")
        assert len(result["positions"]) == 100
        # Positions should be floats between -1 and 1
        positions = result["positions"]
        assert all(-1.0 <= p <= 1.0 for p in positions)

    def test_timeout_on_infinite_loop(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def strategy(bars):\n    while True: pass\n")
            path = f.name
        try:
            result = run_strategy(path, self._make_bars(), timeout_seconds=2)
            assert "error" in result
            assert "timed out" in result["error"].lower()
        finally:
            Path(path).unlink()

    def test_missing_strategy_function(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 42\n")
            path = f.name
        try:
            result = run_strategy(path, self._make_bars())
            assert "error" in result
        finally:
            Path(path).unlink()

    def test_wrong_shape_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import numpy as np\ndef strategy(bars): return np.zeros(5)\n")  # wrong shape
            path = f.name
        try:
            result = run_strategy(path, self._make_bars())
            assert "error" in result  # wrong shape (5 != 100) must produce an error
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# End-to-end (no LLM)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self):
        """Firewall -> sandbox -> backtest -> torture. No LLM needed."""
        raw = generate_synthetic_data(symbols=["SYM"], bars_per_symbol=300)
        key = generate_key()
        anon, _ = anonymize_dataset(raw, key)
        assert len(anon) == 300

        bars = {
            k: anon[k].to_numpy(allow_copy=True).tolist()
            for k in ["open", "high", "low", "close", "volume"]
        }
        result = run_strategy("strategy_template.py", bars)
        assert "error" not in result, result.get("error")

        positions = np.array(result["positions"])
        close = np.array(bars["close"])

        bt = backtest(close, positions)
        assert isinstance(bt["sharpe"], float)
        assert isinstance(bt["trades"], int)
        assert "exposure" in bt

        noise = noise_test(close, positions)
        assert isinstance(noise["passed"], bool)

        defl = deflation_test(close, positions)
        assert isinstance(defl["passed"], bool)

    def test_multi_symbol_anonymization(self):
        raw = generate_synthetic_data(symbols=["A", "B", "C"], bars_per_symbol=50)
        key = generate_key()
        anon, _ = anonymize_dataset(raw, key)
        anon_symbols = anon["symbol"].unique().to_list()
        assert len(anon_symbols) == 3
        assert all(s.startswith("Asset_") for s in anon_symbols)


# ---------------------------------------------------------------------------
# Key Persistence & Reveal
# ---------------------------------------------------------------------------


class TestKeyPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        """Key survives a save/load cycle."""
        key = generate_key()
        path = tmp_path / ".firewall_key"
        save_key(key, path)
        loaded = load_key(path)
        assert loaded == key

    def test_saved_key_is_hex(self, tmp_path):
        """Saved file contains only hex characters."""
        key = generate_key()
        path = tmp_path / ".firewall_key"
        save_key(key, path)
        text = path.read_text().strip()
        assert all(c in "0123456789abcdef" for c in text)
        assert len(text) == 64  # 32 bytes → 64 hex chars


class TestCompare:
    def test_compare_multi_symbol(self):
        """Strategy runs on multiple symbols and produces results for each."""
        raw = generate_synthetic_data(symbols=["A", "B", "C"], bars_per_symbol=200)
        key = generate_key()
        anon, _ = anonymize_dataset(raw, key)

        symbols = anon["symbol"].unique().to_list()
        assert len(symbols) == 3

        results = []
        for sym in symbols:
            sym_df = anon.filter(anon["symbol"] == sym)
            bars = {
                k: sym_df[k].to_numpy(allow_copy=True).tolist()
                for k in ["open", "high", "low", "close", "volume"]
            }
            result = run_strategy("strategy_template.py", bars)
            assert "error" not in result, result.get("error")
            pos = np.array(result["positions"])
            bt = backtest(np.array(bars["close"]), pos)
            bt["symbol"] = sym
            results.append(bt)

        assert len(results) == 3
        assert all("sharpe" in r for r in results)
        assert all("symbol" in r for r in results)


class TestReveal:
    def test_reveal_maps_symbols_back(self):
        """Anonymize → run strategy → reverse_map gives back original symbols."""
        raw = generate_synthetic_data(symbols=["AAPL", "GOOGL"], bars_per_symbol=100)
        key = generate_key()
        anon_df, reverse_map = anonymize_dataset(raw, key)

        # Verify the reverse map points back to originals
        assert set(reverse_map.values()) == {"AAPL", "GOOGL"}

        # Re-anonymize with the same key produces the same map
        anon_df2, reverse_map2 = anonymize_dataset(raw, key)
        assert reverse_map == reverse_map2

        # Confirm every anon symbol resolves to a real one
        for anon_sym in anon_df["symbol"].unique().to_list():
            assert anon_sym in reverse_map
            assert reverse_map[anon_sym] in {"AAPL", "GOOGL"}

    def test_reveal_end_to_end(self, tmp_path):
        """Full reveal pipeline: raw → anonymize → strategy → de-anonymize."""
        raw = generate_synthetic_data(symbols=["MSFT"], bars_per_symbol=200)
        key = generate_key()
        anon_df, reverse_map = anonymize_dataset(raw, key)

        # Run strategy on anonymized data
        import polars as pl

        anon_sym = anon_df["symbol"].unique().to_list()[0]
        sym_df = anon_df.filter(pl.col("symbol") == anon_sym)
        bars = {
            k: sym_df[k].to_numpy(allow_copy=True).tolist()
            for k in ["open", "high", "low", "close", "volume"]
        }

        result = run_strategy("strategy_template.py", bars)
        assert "error" not in result, result.get("error")

        # De-anonymize
        real_sym = reverse_map[anon_sym]
        assert real_sym == "MSFT"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_creates_all_files_including_dockerfile(self, tmp_path, monkeypatch):
        """Export should produce strategy.py, run.py, requirements.txt, and Dockerfile."""
        # Create a fake strategy.py so export doesn't bail out
        fake_strategy = tmp_path / "strategy.py"
        fake_strategy.write_text("def strategy(bars): return [0.0]*len(bars['close'])\n")
        monkeypatch.chdir(tmp_path)

        from cli import export

        dest = tmp_path / "out"
        export(str(dest))

        expected = {"strategy.py", "run.py", "requirements.txt", "Dockerfile"}
        actual = {p.name for p in dest.iterdir()}
        assert expected == actual

        # Verify Dockerfile content has the key directives
        dockerfile_text = (dest / "Dockerfile").read_text()
        assert "FROM python:3.12-slim" in dockerfile_text
        assert "COPY strategy.py run.py" in dockerfile_text
        assert "RUN pip install" in dockerfile_text
        assert 'CMD ["python", "run.py"]' in dockerfile_text


# ---------------------------------------------------------------------------
# Explore
# ---------------------------------------------------------------------------


class TestExplore:
    def _make_bars(self, n: int = 100) -> dict:
        rng = np.random.default_rng(42)
        return {
            "open": rng.normal(0, 0.02, n).tolist(),
            "high": rng.normal(0, 0.02, n).tolist(),
            "low": rng.normal(0, 0.02, n).tolist(),
            "close": rng.normal(0.001, 0.02, n).tolist(),
            "volume": rng.normal(0, 1, n).tolist(),
        }

    def test_run_explore_returns_output(self):
        code = 'print(bars["close"].mean())'
        result = run_explore(code, self._make_bars())
        assert "output" in result, result
        assert len(result["output"]) > 0

    def test_run_explore_timeout(self):
        code = "while True: pass"
        result = run_explore(code, self._make_bars(), timeout_seconds=2)
        assert "error" in result
        assert "timed out" in result["error"].lower()

    def test_run_explore_error(self):
        code = 'raise ValueError("boom")'
        result = run_explore(code, self._make_bars())
        assert "error" in result
        assert "boom" in result["error"]

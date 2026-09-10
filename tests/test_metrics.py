"""Unit tests for the Prometheus-style metrics registry."""

from __future__ import annotations

import time

from alpha_agent.metrics import Metrics


class TestMetrics:
    def test_counter(self) -> None:
        m = Metrics()
        m.inc("requests_total")
        m.inc("requests_total")
        out = m.render()
        assert "alpha_agent_requests_total 2" in out

    def test_histogram_quantiles(self) -> None:
        m = Metrics()
        for i in range(1, 101):
            m.observe("req_duration", float(i) / 100.0)
        out = m.render()
        assert "alpha_agent_req_duration_count 100" in out
        # p50 ~0.5
        assert 'alpha_agent_req_duration_quantile{quantile="0.5"}' in out

    def test_empty_render(self) -> None:
        m = Metrics()
        out = m.render()
        assert out.endswith("\n")

    def test_timer_context(self) -> None:
        m = Metrics()
        with m.time("llm"):
            time.sleep(0.001)
        out = m.render()
        assert "alpha_agent_llm_count 1" in out

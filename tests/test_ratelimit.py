"""Unit tests for the token-bucket rate limiter."""

from __future__ import annotations

from alpha_agent.ratelimit import RateLimiter, TokenBucket


class TestTokenBucket:
    def test_full_capacity_allows_burst(self) -> None:
        bucket = TokenBucket(capacity=5, refill_rate=0.0)
        assert all(bucket.consume() for _ in range(5))
        assert not bucket.consume()  # exhausted

    def test_remaining_reports_correctly(self) -> None:
        bucket = TokenBucket(capacity=10, refill_rate=0.0)
        bucket.consume()
        assert bucket.remaining == 9

    def test_refill_over_time(self) -> None:
        bucket = TokenBucket(capacity=1, refill_rate=60.0)  # 1 token/sec
        assert bucket.consume()
        assert not bucket.consume()
        # simulate 50ms of refill — 60/sec = 3 tokens in 50ms? no: 60/s = 0.06/ms → 50ms → 0.3
        import time
        time.sleep(0.02)  # 20ms → 1.2 tokens (>=1)
        assert bucket.consume()


class TestRateLimiter:
    def test_limits_per_key(self) -> None:
        limiter = RateLimiter(default_per_minute=3)
        key = "aa_test"
        allowed = [limiter.check(key)[0] for _ in range(4)]
        assert allowed == [True, True, True, False]

    def test_independent_keys(self) -> None:
        limiter = RateLimiter(default_per_minute=2)
        # k1: consume 2 of 2 → exhausted
        assert limiter.check("k1")[0] is True
        assert limiter.check("k1")[0] is True
        assert limiter.check("k1")[0] is False
        # k2: consume 2 of 2 → exhausted (independent of k1)
        assert limiter.check("k2")[0] is True
        assert limiter.check("k2")[0] is True
        assert limiter.check("k2")[0] is False

    def test_custom_per_minute(self) -> None:
        limiter = RateLimiter(default_per_minute=60)
        limiter.check("k1", per_minute=1)
        assert limiter.check("k1", per_minute=1)[0] is False

"""Offline checks of LinkedIn's request pacing and 429 retries (no network)."""

from types import SimpleNamespace

import pytest

import jobspy.linkedin as li
from jobspy.linkedin import LinkedIn
from jobspy.linkedin.util import retry_after


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeSession:
    def __init__(self, statuses: list[int], headers: dict | None = None) -> None:
        self.statuses = list(statuses)
        self.headers = headers or {}
        self.calls = 0
        self.rotates = False
        self.cookies = SimpleNamespace(clear=lambda: None)

    def get(self, url: str, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        assert kwargs.get("allow_redirects") is True
        return SimpleNamespace(status_code=self.statuses.pop(0), headers=self.headers)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(li.time, "monotonic", c.monotonic)
    monkeypatch.setattr(li.time, "sleep", c.sleep)
    monkeypatch.setattr(li.random, "uniform", lambda a, b: a)
    return c


def scraper(session: FakeSession) -> LinkedIn:
    s = LinkedIn()
    s.session = session  # type: ignore[assignment]
    s.next_request = 0.0
    s.slowdown = 1.0
    return s


def test_429_is_retried_with_doubling_backoff(clock: Clock) -> None:
    session = FakeSession([429, 429, 200])
    assert scraper(session)._get("u").status_code == 200
    assert session.calls == 3
    assert clock.sleeps == [5.0, 10.0]


def test_gives_up_after_max_retries(clock: Clock) -> None:
    session = FakeSession([429] * 5)
    assert scraper(session)._get("u").status_code == 429
    assert session.calls == 5
    assert clock.sleeps == [5.0, 10.0, 20.0, 40.0]


def test_retry_after_header_wins(clock: Clock) -> None:
    session = FakeSession([429, 200], {"retry-after": "12"})
    scraper(session)._get("u")
    assert clock.sleeps == [12.0]


def test_consecutive_requests_are_paced(clock: Clock) -> None:
    s = scraper(FakeSession([200, 200, 404]))
    s._get("a")
    s._get("b")
    assert s._get("c").status_code == 404  # not retried
    assert clock.sleeps == [2.5, 2.5]


@pytest.mark.parametrize(
    "value, seconds",
    [("7", 7.0), (None, None), ("Wed, 21 Oct 2015 07:28:00 GMT", None)],
)
def test_retry_after(value: str | None, seconds: float | None) -> None:
    assert retry_after(value) == seconds


def test_429_slows_the_pace_and_success_eases_it(clock: Clock) -> None:
    s = scraper(FakeSession([429, 200, 200, 200]))
    s._get("a")  # 429 then 200: 1.5, then 1.35
    assert s.slowdown == pytest.approx(1.35)
    s._get("b")
    s._get("c")
    assert clock.sleeps[1:] == pytest.approx([2.5 * 1.35, 2.5 * 1.215])

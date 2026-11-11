"""List prices for cost estimates.

Prices are USD per 1M tokens on the paid tier, text input. Thinking tokens
are billed as output. On the Gemini free tier the real cost is $0, but we
still record the list price: it shows what the same traffic would cost in
production.

Source: https://aistudio.google.com/docs/pricing (checked 2026-11-11).
"""

from typing import NamedTuple

from docintel.llm.types import Usage


class Price(NamedTuple):
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, Price] = {
    "gemini-3.5-flash-lite": Price(0.30, 2.50),
    "gemini-3.1-flash-lite": Price(0.25, 1.50),
    "gemini-2.5-flash-lite": Price(0.10, 0.40),
    # Introductory price through 2026-12-31; doubles from 2027-01-01.
    "gemini-3.8-flash": Price(0.75, 3.75),
    "fake": Price(0.0, 0.0),
}


def estimate_cost(model: str, usage: Usage) -> float | None:
    """Return the estimated cost in USD, or None if the model is not priced."""
    price = PRICES.get(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000

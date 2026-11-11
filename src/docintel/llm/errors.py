"""Provider-neutral exceptions.

Adapters translate their SDK errors into these, so the rest of the code
never imports a provider SDK just to catch an exception.
"""


class LLMError(Exception):
    """Base class for all LLM layer errors."""

    retryable: bool = False


class LLMRateLimitError(LLMError):
    """The provider returned 429: too many requests or quota exhausted."""

    retryable = True


class LLMUnavailableError(LLMError):
    """Temporary provider or network failure (5xx, connection reset)."""

    retryable = True


class LLMTimeoutError(LLMError):
    """The call did not finish within the configured timeout."""

    retryable = True


class LLMRequestError(LLMError):
    """The request itself is wrong (bad key, bad model, invalid input).

    Retrying will not help, so this is never retried.
    """

"""Retry logic with exponential backoff for API calls."""

import asyncio
import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, TypeVar, Any, Optional, Tuple, Type
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)


def calculate_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool
) -> float:
    """
    Calculate delay for a retry attempt using exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    delay = initial_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)

    if jitter:
        # Add jitter: random value between 0 and delay
        delay = delay * (0.5 + random.random())

    return delay


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        config: Retry configuration
        on_retry: Optional callback called on each retry with (exception, attempt)

    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    # Check if it's a rate limit or retryable status code
                    status_code = getattr(e, "status_code", None)
                    if status_code and status_code not in config.retryable_status_codes:
                        raise

                    if attempt < config.max_retries:
                        delay = calculate_delay(
                            attempt,
                            config.initial_delay,
                            config.max_delay,
                            config.exponential_base,
                            config.jitter
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_retries + 1} attempts failed. Last error: {e}"
                        )

            raise last_exception

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    # Check if it's a rate limit or retryable status code
                    status_code = getattr(e, "status_code", None)
                    if status_code and status_code not in config.retryable_status_codes:
                        raise

                    if attempt < config.max_retries:
                        delay = calculate_delay(
                            attempt,
                            config.initial_delay,
                            config.max_delay,
                            config.exponential_base,
                            config.jitter
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_retries + 1} attempts failed. Last error: {e}"
                        )

            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class RetryableError(Exception):
    """Exception that should trigger a retry."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(RetryableError):
    """Rate limit exceeded error."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, provider: str, message: str):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class AuthenticationError(ProviderError):
    """Authentication failed error."""
    pass


class APIError(ProviderError):
    """General API error."""

    def __init__(self, provider: str, message: str, status_code: Optional[int] = None):
        super().__init__(provider, message)
        self.status_code = status_code

"""
Offline-aware utilities for HuggingFace and Transformers.

Provides safe wrappers around from_pretrained() that try online loading first,
then fall back to local-only mode when the network is unavailable.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Exception types that indicate network/connectivity failures.
# OSError covers: "Errno -2: Name or service not known", DNS failures, etc.
# huggingface_hub internally raises requests exceptions which are subclasses of
# ConnectionError / TimeoutError when the network is unreachable.
NETWORK_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def safe_from_pretrained(model_class: Any, *args: Any, **kwargs: Any) -> Any:
    """Try model_class.from_pretrained() normally, fall back to local_files_only on network error.

    Strategy:
      1. Attempt the normal from_pretrained call (online + cache).
      2. On a network-related exception, retry with local_files_only=True so only
         the local HuggingFace cache is consulted.
      3. If the retry also fails (model was never cached), the exception propagates
         with full context so callers get a clear error message.

    Args:
        model_class: A HuggingFace class with a from_pretrained classmethod
                     (e.g. CLIPTokenizer, AutoTokenizer, AutoModelForCausalLM).
        *args: Positional arguments forwarded to from_pretrained.
        **kwargs: Keyword arguments forwarded to from_pretrained. If 'local_files_only'
                  is already set, the fallback is skipped.

    Returns:
        The object returned by from_pretrained.

    Raises:
        Any exception from the final from_pretrained call if offline fallback
        also fails.
    """
    if kwargs.get("local_files_only"):
        # Caller explicitly requested local-only; no point in the try-online path.
        return model_class.from_pretrained(*args, **kwargs)

    try:
        return model_class.from_pretrained(*args, **kwargs)
    except NETWORK_EXCEPTIONS as e:
        logger.warning(
            "HuggingFace connection failed for %s, retrying with local_files_only=True: %s",
            model_class.__name__,
            e,
        )
        kwargs["local_files_only"] = True
        return model_class.from_pretrained(*args, **kwargs)

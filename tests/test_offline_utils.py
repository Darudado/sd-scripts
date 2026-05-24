"""Tests for library.offline_utils — offline fallback for HuggingFace models."""

import pytest
from unittest.mock import MagicMock, patch

from library.offline_utils import NETWORK_EXCEPTIONS, safe_from_pretrained


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeModelClass:
    """Minimal stand-in for a HuggingFace model class with from_pretrained."""

    call_count = 0
    last_kwargs = {}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.call_count += 1
        cls.last_kwargs = kwargs
        return f"model_instance_{cls.call_count}"


class FakeModelClassNetworkError(FakeModelClass):
    """Always raises a network error, then succeeds on local_files_only retry."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.call_count += 1
        cls.last_kwargs = kwargs
        if not kwargs.get("local_files_only"):
            raise ConnectionError("No route to host")
        return "cached_model_instance"


class FakeModelClassAlwaysFails(FakeModelClass):
    """Always raises — simulates model never cached."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.call_count += 1
        cls.last_kwargs = kwargs
        if not kwargs.get("local_files_only"):
            raise ConnectionError("No route to host")
        raise OSError("offline and not cached")


class FakeModelClassOtherError(FakeModelClass):
    """Raises a non-network error (e.g. missing key)."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.call_count += 1
        raise ValueError("some model config error")


class FakeModelClassAlreadyLocal(FakeModelClass):
    """Raises on first call to verify local_files_only short-circuit."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.call_count += 1
        if kwargs.get("local_files_only"):
            return "local_model"
        raise ConnectionError("should not happen when local_files_only pre-set")


# ---------------------------------------------------------------------------
# tests — NETWORK_EXCEPTIONS
# ---------------------------------------------------------------------------

class TestNetworkExceptions:
    def test_contains_connection_error(self):
        assert ConnectionError in NETWORK_EXCEPTIONS

    def test_contains_timeout_error(self):
        assert TimeoutError in NETWORK_EXCEPTIONS

    def test_contains_os_error(self):
        assert OSError in NETWORK_EXCEPTIONS

    def test_subclass_of_os_error_is_caught(self):
        """DNS failures raise OSError subclasses like socket.gaierror."""
        with pytest.raises(OSError):
            raise OSError("[Errno -2] Name or service not known")


# ---------------------------------------------------------------------------
# tests — safe_from_pretrained (online success)
# ---------------------------------------------------------------------------

class TestSafeFromPretrainedOnlineSuccess:
    def test_returns_model_on_success(self):
        FakeModelClass.call_count = 0
        result = safe_from_pretrained(FakeModelClass, "some-model-id")
        assert result == "model_instance_1"

    def test_passes_args_and_kwargs(self):
        FakeModelClass.call_count = 0
        safe_from_pretrained(
            FakeModelClass, "model-id", subfolder="tokenizer", torch_dtype="float16"
        )
        assert FakeModelClass.last_kwargs == {
            "subfolder": "tokenizer",
            "torch_dtype": "float16",
        }

    def test_only_one_call_on_success(self):
        FakeModelClass.call_count = 0
        safe_from_pretrained(FakeModelClass, "model-id")
        assert FakeModelClass.call_count == 1


# ---------------------------------------------------------------------------
# tests — safe_from_pretrained (network error → fallback to local)
# ---------------------------------------------------------------------------

class TestSafeFromPretrainedNetworkFallback:
    def test_falls_back_to_local_on_connection_error(self):
        FakeModelClassNetworkError.call_count = 0
        result = safe_from_pretrained(FakeModelClassNetworkError, "model-id")
        assert result == "cached_model_instance"
        assert FakeModelClassNetworkError.call_count == 2

    def test_sets_local_files_only_on_retry(self):
        FakeModelClassNetworkError.call_count = 0
        safe_from_pretrained(FakeModelClassNetworkError, "model-id")
        assert FakeModelClassNetworkError.last_kwargs.get("local_files_only") is True

    def test_raises_on_timeout_error(self):
        class TimeoutModel(FakeModelClassNetworkError):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.call_count += 1
                cls.last_kwargs = kwargs
                if not kwargs.get("local_files_only"):
                    raise TimeoutError("timed out")
                return "cached"

        TimeoutModel.call_count = 0
        result = safe_from_pretrained(TimeoutModel, "model-id")
        assert result == "cached"
        assert TimeoutModel.call_count == 2

    def test_raises_on_os_error(self):
        class DNSFailModel(FakeModelClassNetworkError):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.call_count += 1
                cls.last_kwargs = kwargs
                if not kwargs.get("local_files_only"):
                    raise OSError("[Errno -2] Name or service not known")
                return "cached"

        DNSFailModel.call_count = 0
        result = safe_from_pretrained(DNSFailModel, "model-id")
        assert result == "cached"


# ---------------------------------------------------------------------------
# tests — safe_from_pretrained (both attempts fail)
# ---------------------------------------------------------------------------

class TestSafeFromPretrainedBothFail:
    def test_raises_when_local_also_fails(self):
        FakeModelClassAlwaysFails.call_count = 0
        with pytest.raises(OSError, match="offline and not cached"):
            safe_from_pretrained(FakeModelClassAlwaysFails, "model-id")
        assert FakeModelClassAlwaysFails.call_count == 2

    def test_non_network_error_propagates_immediately(self):
        """A ValueError (e.g. model config error) should NOT trigger retry."""
        FakeModelClassOtherError.call_count = 0
        with pytest.raises(ValueError, match="some model config error"):
            safe_from_pretrained(FakeModelClassOtherError, "model-id")
        assert FakeModelClassOtherError.call_count == 1


# ---------------------------------------------------------------------------
# tests — local_files_only already set
# ---------------------------------------------------------------------------

class TestSafeFromPretrainedLocalOnly:
    def test_skips_try_online_when_local_files_only_pre_set(self):
        FakeModelClassAlreadyLocal.call_count = 0
        result = safe_from_pretrained(
            FakeModelClassAlreadyLocal, "model-id", local_files_only=True
        )
        assert result == "local_model"
        assert FakeModelClassAlreadyLocal.call_count == 1

    def test_passes_local_files_only_through_kwargs(self):
        FakeModelClass.call_count = 0
        safe_from_pretrained(FakeModelClass, "model-id", local_files_only=True)
        assert FakeModelClass.last_kwargs.get("local_files_only") is True
        assert FakeModelClass.call_count == 1


# ---------------------------------------------------------------------------
# tests — logging
# ---------------------------------------------------------------------------

class TestSafeFromPretrainedLogging:
    def test_warns_on_fallback(self, caplog):
        """Verify a warning is logged when falling back to local mode."""
        FakeModelClassNetworkError.call_count = 0
        with caplog.at_level("WARNING", logger="library.offline_utils"):
            safe_from_pretrained(FakeModelClassNetworkError, "model-id")
        assert "retrying with local_files_only=True" in caplog.text

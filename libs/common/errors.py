"""Standardized application error model and RFC 7807 problem details conversion.

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why RFC 7807 Problem Details for HTTP APIs?
--------------------------------------------------------------------------------
In modern distributed microservices and multi-agent platforms, error handling
is often inconsistent: some endpoints return `{ "error": "failed" }`, others return
`{ "message": "not found" }`, and unhandled bugs return full Python tracebacks!

RFC 7807 ("Problem Details for HTTP APIs") standardizes error responses:
1. Machine-Readable:
   Clients can branch on machine-readable error codes (e.g., `GOVERNANCE_BLOCKED` or
   `RATE_LIMIT_EXCEEDED`) rather than parsing unstructured English messages.
2. Security & Information Hiding:
   Internal database table structures or stack traces are never exposed to API callers.
3. Extensibility:
   Domain-specific error context (such as required approval tokens or retry durations)
   can be attached cleanly to the `invalid_params` payload.
================================================================================
"""

from typing import Any


class AppError(Exception):
    """Base exception for all domain and platform errors.

    All custom exceptions in the platform inherit from AppError to ensure uniform
    translation into HTTP status codes and RFC 7807 JSON payloads.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_problem_detail(self, instance_path: str | None = None) -> dict[str, Any]:
        """Convert error to RFC 7807 Problem Details representation.

        Args:
            instance_path: Request URI path where the error occurred.

        Returns:
            Dictionary conforming to RFC 7807 problem details specification.
        """
        problem: dict[str, Any] = {
            "type": f"urn:aiops:error:{self.error_code.lower().replace('_', '-')}",
            "title": self.error_code,
            "status": self.status_code,
            "detail": self.message,
            "error_code": self.error_code,
        }
        if self.details:
            problem["invalid_params"] = self.details
        if instance_path:
            problem["instance"] = instance_path
        return problem


class NotFoundError(AppError):
    """Resource not found (HTTP 404).

    Used when a requested collection, document, run, or tool definition does not exist.
    """

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        extra = details or {}
        if resource_type:
            extra["resource_type"] = resource_type
        if resource_id:
            extra["resource_id"] = resource_id
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=404,
            details=extra,
        )


class ValidationError(AppError):
    """Input validation failure (HTTP 422).

    Used when input payloads fail domain validation rules beyond basic schema parsing.
    """

    def __init__(
        self,
        message: str = "Validation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=details,
        )


class AuthenticationError(AppError):
    """Missing or invalid authentication credentials (HTTP 401).

    Used when bearer tokens are missing, expired, or cryptographically invalid.
    """

    def __init__(
        self,
        message: str = "Authentication required or credentials invalid",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            status_code=401,
            details=details,
        )


class AuthorizationError(AppError):
    """Actor lacks permissions for the requested resource (HTTP 403).

    Used when an authenticated user attempts to access another tenant's data or lacks RBAC roles.
    """

    def __init__(
        self,
        message: str = "Permission denied",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_DENIED",
            status_code=403,
            details=details,
        )


class GovernanceError(AppError):
    """Action rejected by governance gate or requires human approval token (HTTP 403).

    Core P0 requirement: High-risk write operations (SQL mutations, external webhooks)
    are strictly halted unless a valid, signed, single-use approval token is present.
    """

    def __init__(
        self,
        message: str = "Action violates governance policy or requires approval token",
        action: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        extra = details or {}
        if action:
            extra["action"] = action
        super().__init__(
            message=message,
            error_code="GOVERNANCE_BLOCKED",
            status_code=403,
            details=extra,
        )


class RateLimitError(AppError):
    """Request rate limit or quota exceeded (HTTP 429).

    Signals to callers and retry clients that requests should be throttled.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after_seconds: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        extra = details or {}
        if retry_after_seconds is not None:
            extra["retry_after"] = retry_after_seconds
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=extra,
        )


class ConflictError(AppError):
    """Resource conflict or concurrency violation (HTTP 409).

    Used when attempting to create a resource with a duplicate unique key or version collision.
    """

    def __init__(
        self,
        message: str = "Resource conflict detected",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="RESOURCE_CONFLICT",
            status_code=409,
            details=details,
        )


class ExternalServiceError(AppError):
    """Downstream service, LLM provider, or MCP server failed (HTTP 502).

    Distinguishes internal platform bugs (500) from external provider outages (502).
    """

    def __init__(
        self,
        message: str = "External service unavailable or returned error",
        service_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        extra = details or {}
        if service_name:
            extra["service_name"] = service_name
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details=extra,
        )

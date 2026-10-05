"""Standardized application error model and RFC 7807 problem details conversion."""

from typing import Any


class AppError(Exception):
    """Base exception for all domain and platform errors."""

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
        """Convert error to RFC 7807 Problem Details representation."""
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
    """Resource not found."""

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
    """Input validation failure."""

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
    """Missing or invalid authentication credentials."""

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
    """Actor lacks permissions for the requested resource."""

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
    """Action rejected by governance gate or requires missing human approval token."""

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
    """Request rate limit or quota exceeded."""

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
    """Resource conflict or concurrency violation."""

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
    """Downstream service, LLM provider, or MCP server failed."""

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

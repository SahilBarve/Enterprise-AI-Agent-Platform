"""Unit tests for typed exceptions and RFC 7807 error models."""

import pytest

from libs.common.errors import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    ExternalServiceError,
    GovernanceError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)


@pytest.mark.unit
def test_app_error_rfc7807_serialization() -> None:
    """Validate base AppError serialization to RFC 7807 problem details."""
    error = AppError(
        message="A database constraint failed",
        error_code="DB_CONSTRAINT_VIOLATION",
        status_code=400,
        details={"field": "email", "issue": "already exists"},
    )
    problem = error.to_problem_detail(instance_path="/api/v1/users")

    assert problem["type"] == "urn:aiops:error:db-constraint-violation"
    assert problem["title"] == "DB_CONSTRAINT_VIOLATION"
    assert problem["status"] == 400
    assert problem["detail"] == "A database constraint failed"
    assert problem["error_code"] == "DB_CONSTRAINT_VIOLATION"
    assert problem["invalid_params"] == {"field": "email", "issue": "already exists"}
    assert problem["instance"] == "/api/v1/users"


@pytest.mark.unit
def test_not_found_error() -> None:
    """Validate NotFoundError defaults and metadata."""
    error = NotFoundError(
        message="Document not found",
        resource_type="document",
        resource_id="doc-123",
    )
    assert error.status_code == 404
    assert error.error_code == "RESOURCE_NOT_FOUND"
    assert error.details["resource_type"] == "document"
    assert error.details["resource_id"] == "doc-123"


@pytest.mark.unit
def test_validation_error() -> None:
    """Validate ValidationError status code and details."""
    error = ValidationError(
        message="Invalid payload",
        details={"query": "cannot be empty"},
    )
    assert error.status_code == 422
    assert error.error_code == "VALIDATION_ERROR"
    assert error.details["query"] == "cannot be empty"


@pytest.mark.unit
def test_governance_error() -> None:
    """Validate GovernanceError for unapproved mutations."""
    error = GovernanceError(
        message="SQL mutation requires human approval token",
        action="execute_sql_mutation",
        details={"run_id": "run-456"},
    )
    assert error.status_code == 403
    assert error.error_code == "GOVERNANCE_BLOCKED"
    assert error.details["action"] == "execute_sql_mutation"
    assert error.details["run_id"] == "run-456"


@pytest.mark.unit
def test_rate_limit_error() -> None:
    """Validate RateLimitError status code and retry_after header detail."""
    error = RateLimitError(
        message="Too many requests",
        retry_after_seconds=30,
    )
    assert error.status_code == 429
    assert error.error_code == "RATE_LIMIT_EXCEEDED"
    assert error.details["retry_after"] == 30


@pytest.mark.unit
def test_auth_errors() -> None:
    """Validate AuthenticationError and AuthorizationError."""
    authn = AuthenticationError("JWT expired")
    assert authn.status_code == 401
    assert authn.error_code == "AUTHENTICATION_FAILED"

    authz = AuthorizationError("Insufficient permissions")
    assert authz.status_code == 403
    assert authz.error_code == "AUTHORIZATION_DENIED"


@pytest.mark.unit
def test_conflict_and_external_errors() -> None:
    """Validate ConflictError and ExternalServiceError."""
    conflict = ConflictError("Resource already locked")
    assert conflict.status_code == 409
    assert conflict.error_code == "RESOURCE_CONFLICT"

    external = ExternalServiceError(
        message="Tavily API timeout",
        service_name="tavily",
    )
    assert external.status_code == 502
    assert external.error_code == "EXTERNAL_SERVICE_ERROR"
    assert external.details["service_name"] == "tavily"

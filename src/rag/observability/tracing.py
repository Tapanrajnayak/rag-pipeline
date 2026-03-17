"""OpenTelemetry tracing setup and span helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode


def configure_tracing(
    service_name: str,
    otlp_endpoint: str | None = None,
) -> None:
    """Configure OpenTelemetry tracing.

    Args:
        service_name: Service name tag for traces.
        otlp_endpoint: OTLP gRPC endpoint (e.g. 'http://localhost:4317').
                       If None, tracing is configured with a no-op exporter.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass  # opentelemetry-exporter-otlp not installed

    trace.set_tracer_provider(provider)


def get_tracer(name: str) -> trace.Tracer:
    """Return a named tracer."""
    return trace.get_tracer(name)


@contextmanager
def span(
    tracer: trace.Tracer,
    name: str,
    attributes: dict[str, str | int | float | bool] | None = None,
) -> Generator[Span, None, None]:
    """Context manager for a named span with optional attributes.

    Sets span status to ERROR automatically on exception.

    Args:
        tracer: OTel tracer instance.
        name: Span name.
        attributes: Optional key-value attributes.

    Yields:
        Active span.
    """
    with tracer.start_as_current_span(name) as current_span:
        if attributes:
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
        try:
            yield current_span
        except Exception as exc:
            current_span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

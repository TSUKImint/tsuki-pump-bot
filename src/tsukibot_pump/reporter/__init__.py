"""Performance reporter.

Reads the SQLite event store + the paper-fill diagnostic fields and
emits a per-cohort summary so you can see *which filter / score bucket
/ exit reason* is producing P&L versus burning fees.
"""

from .report import (
    AggregateReport,
    BucketStats,
    EventCounts,
    ExitReasonStats,
    FeeBreakdown,
    LatencyStats,
    ScoreBucket,
    build_report,
    format_report_text,
)

__all__ = [
    "AggregateReport",
    "BucketStats",
    "EventCounts",
    "ExitReasonStats",
    "FeeBreakdown",
    "LatencyStats",
    "ScoreBucket",
    "build_report",
    "format_report_text",
]

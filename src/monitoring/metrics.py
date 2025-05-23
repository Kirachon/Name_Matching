"""
Metrics collection for monitoring name matching performance.
"""

import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and exposes metrics for monitoring."""
    
    def __init__(self):
        """Initialize metrics collector."""
        # Internal metrics storage
        self._metrics = {
            "match_requests_total": 0,
            "match_requests_successful": 0,
            "match_requests_failed": 0,
            "processing_time_total": 0.0,
        }
    
    def record_match_request(self, status: str, processing_time: float):
        """Record a name matching request."""
        self._metrics["match_requests_total"] += 1
        
        if status == "success":
            self._metrics["match_requests_successful"] += 1
        else:
            self._metrics["match_requests_failed"] += 1
        
        self._metrics["processing_time_total"] += processing_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = self._metrics.copy()
        
        # Calculate derived metrics
        total_requests = metrics["match_requests_total"]
        if total_requests > 0:
            metrics["success_rate"] = metrics["match_requests_successful"] / total_requests
            metrics["average_processing_time"] = metrics["processing_time_total"] / total_requests
        else:
            metrics["success_rate"] = 0.0
            metrics["average_processing_time"] = 0.0
        
        return metrics
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
            processing_time = time.time() - start_time
            self.record_match_request("success", processing_time)
        except Exception as e:
            processing_time = time.time() - start_time
            self.record_match_request("error", processing_time)
            raise


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector

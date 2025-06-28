"""
Structured Logging Utilities for Ragamala Painting Generation.

This module provides comprehensive structured logging functionality with support for
JSON formatting, multiple output destinations, contextual logging, and integration
with monitoring systems for the Ragamala painting generation project.
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import threading
import traceback
from contextlib import contextmanager
import uuid

# Third-party imports
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install
    RICH_AVAILABLE = True
    install(show_locals=True)
except ImportError:
    RICH_AVAILABLE = False

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class LogConfig:
    """Configuration for logging setup."""
    # Basic settings
    name: str = "ragamala"
    level: Union[str, int] = logging.INFO
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    # Console logging
    enable_console: bool = True
    console_level: Union[str, int] = logging.INFO
    use_rich_console: bool = True
    
    # File logging
    enable_file: bool = True
    file_level: Union[str, int] = logging.DEBUG
    log_file: Optional[str] = None
    log_dir: str = "logs"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # JSON logging
    enable_json: bool = True
    json_file: Optional[str] = None
    json_level: Union[str, int] = logging.DEBUG
    
    # Structured logging
    enable_structured: bool = True
    include_caller_info: bool = True
    include_process_info: bool = True
    include_thread_info: bool = True
    
    # Performance
    async_logging: bool = False
    buffer_size: int = 1000
    
    # Monitoring integration
    enable_metrics: bool = False
    metrics_endpoint: Optional[str] = None

class ContextualFilter(logging.Filter):
    """Filter to add contextual information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
        self.thread_local = threading.local()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add contextual information to the log record."""
        # Add global context
        for key, value in self.context.items():
            setattr(record, key, value)
        
        # Add thread-local context
        thread_context = getattr(self.thread_local, 'context', {})
        for key, value in thread_context.items():
            setattr(record, key, value)
        
        # Add caller information
        if hasattr(record, 'pathname'):
            record.module_name = Path(record.pathname).stem
        
        # Add unique request ID if not present
        if not hasattr(record, 'request_id'):
            record.request_id = str(uuid.uuid4())[:8]
        
        return True
    
    def set_context(self, **kwargs):
        """Set thread-local context."""
        if not hasattr(self.thread_local, 'context'):
            self.thread_local.context = {}
        self.thread_local.context.update(kwargs)
    
    def clear_context(self):
        """Clear thread-local context."""
        if hasattr(self.thread_local, 'context'):
            self.thread_local.context.clear()

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, 
                 include_caller_info: bool = True,
                 include_process_info: bool = True,
                 include_thread_info: bool = True):
        super().__init__()
        self.include_caller_info = include_caller_info
        self.include_process_info = include_process_info
        self.include_thread_info = include_thread_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add caller information
        if self.include_caller_info:
            log_entry.update({
                'filename': record.filename,
                'lineno': record.lineno,
                'funcName': record.funcName,
                'module': getattr(record, 'module_name', record.module),
            })
        
        # Add process information
        if self.include_process_info:
            log_entry.update({
                'process': record.process,
                'processName': record.processName,
            })
        
        # Add thread information
        if self.include_thread_info:
            log_entry.update({
                'thread': record.thread,
                'threadName': record.threadName,
            })
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info', 'message'
            }:
                try:
                    json.dumps(value)  # Test if serializable
                    extra_fields[key] = value
                except (TypeError, ValueError):
                    extra_fields[key] = str(value)
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, ensure_ascii=False)

class MetricsHandler(logging.Handler):
    """Handler to send log metrics to monitoring systems."""
    
    def __init__(self, metrics_endpoint: Optional[str] = None):
        super().__init__()
        self.metrics_endpoint = metrics_endpoint
        self.metrics_buffer = []
        self.buffer_lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord):
        """Emit log record as metric."""
        try:
            metric = {
                'timestamp': record.created,
                'level': record.levelname,
                'logger': record.name,
                'message_length': len(record.getMessage()),
            }
            
            with self.buffer_lock:
                self.metrics_buffer.append(metric)
                
                # Flush buffer if it gets too large
                if len(self.metrics_buffer) >= 100:
                    self._flush_metrics()
        
        except Exception:
            self.handleError(record)
    
    def _flush_metrics(self):
        """Flush metrics buffer to monitoring system."""
        if self.metrics_endpoint and self.metrics_buffer:
            # Implementation would depend on specific monitoring system
            # For now, just clear the buffer
            self.metrics_buffer.clear()

class StructuredLogger:
    """Main structured logger class."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.logger = None
        self.contextual_filter = ContextualFilter()
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger with all configured handlers."""
        # Create logger
        self.logger = logging.getLogger(self.config.name)
        self.logger.setLevel(self.config.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add contextual filter
        self.logger.addFilter(self.contextual_filter)
        
        # Setup console handler
        if self.config.enable_console:
            self._setup_console_handler()
        
        # Setup file handler
        if self.config.enable_file:
            self._setup_file_handler()
        
        # Setup JSON handler
        if self.config.enable_json:
            self._setup_json_handler()
        
        # Setup metrics handler
        if self.config.enable_metrics:
            self._setup_metrics_handler()
    
    def _setup_console_handler(self):
        """Setup console handler with optional Rich formatting."""
        if RICH_AVAILABLE and self.config.use_rich_console:
            handler = RichHandler(
                console=Console(stderr=True),
                show_path=True,
                show_time=True,
                rich_tracebacks=True,
                tracebacks_show_locals=True
            )
        else:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(self.config.format_string)
            handler.setFormatter(formatter)
        
        handler.setLevel(self.config.console_level)
        self.logger.addHandler(handler)
    
    def _setup_file_handler(self):
        """Setup rotating file handler."""
        # Create log directory
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine log file path
        if self.config.log_file:
            log_file = log_dir / self.config.log_file
        else:
            log_file = log_dir / f"{self.config.name}.log"
        
        # Create rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(self.config.format_string)
        handler.setFormatter(formatter)
        handler.setLevel(self.config.file_level)
        
        self.logger.addHandler(handler)
    
    def _setup_json_handler(self):
        """Setup JSON file handler for structured logging."""
        # Create log directory
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine JSON log file path
        if self.config.json_file:
            json_file = log_dir / self.config.json_file
        else:
            json_file = log_dir / f"{self.config.name}.jsonl"
        
        # Create rotating file handler for JSON
        handler = logging.handlers.RotatingFileHandler(
            json_file,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        
        # Use custom JSON formatter
        formatter = JSONFormatter(
            include_caller_info=self.config.include_caller_info,
            include_process_info=self.config.include_process_info,
            include_thread_info=self.config.include_thread_info
        )
        handler.setFormatter(formatter)
        handler.setLevel(self.config.json_level)
        
        self.logger.addHandler(handler)
    
    def _setup_metrics_handler(self):
        """Setup metrics handler for monitoring integration."""
        handler = MetricsHandler(self.config.metrics_endpoint)
        handler.setLevel(logging.WARNING)  # Only send warnings and errors as metrics
        self.logger.addHandler(handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger
    
    def set_context(self, **kwargs):
        """Set logging context."""
        self.contextual_filter.set_context(**kwargs)
    
    def clear_context(self):
        """Clear logging context."""
        self.contextual_filter.clear_context()
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary logging context."""
        old_context = getattr(self.contextual_filter.thread_local, 'context', {}).copy()
        try:
            self.set_context(**kwargs)
            yield
        finally:
            self.contextual_filter.thread_local.context = old_context

class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a named timer."""
        self.timers[name] = time.time()
        self.logger.debug(f"Timer started: {name}")
    
    def end_timer(self, name: str) -> float:
        """End a named timer and log the duration."""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.time() - self.timers[name]
        del self.timers[name]
        
        self.logger.info(f"Timer completed: {name}", extra={
            'timer_name': name,
            'duration_seconds': duration,
            'performance_metric': True
        })
        
        return duration
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            self.logger.debug(f"Starting operation: {name}")
            yield
        finally:
            duration = time.time() - start_time
            self.logger.info(f"Operation completed: {name}", extra={
                'operation_name': name,
                'duration_seconds': duration,
                'performance_metric': True
            })

def setup_logger(name: str = "ragamala", 
                config: Optional[LogConfig] = None) -> logging.Logger:
    """
    Setup a structured logger with the given configuration.
    
    Args:
        name: Logger name
        config: Logging configuration
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = LogConfig(name=name)
    
    structured_logger = StructuredLogger(config)
    return structured_logger.get_logger()

def setup_structlog(config: Optional[LogConfig] = None) -> Any:
    """
    Setup structlog if available.
    
    Args:
        config: Logging configuration
        
    Returns:
        Configured structlog logger
    """
    if not STRUCTLOG_AVAILABLE:
        raise ImportError("structlog is not available. Install with: pip install structlog")
    
    if config is None:
        config = LogConfig()
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if config.enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.processors.KeyValueRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(config.name)

def get_logger(name: str = "ragamala") -> logging.Logger:
    """
    Get or create a logger with default configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with default configuration
    if not logger.handlers:
        config = LogConfig(name=name)
        structured_logger = StructuredLogger(config)
        return structured_logger.get_logger()
    
    return logger

def configure_third_party_loggers(level: Union[str, int] = logging.WARNING):
    """
    Configure third-party library loggers to reduce noise.
    
    Args:
        level: Log level for third-party loggers
    """
    third_party_loggers = [
        'urllib3',
        'requests',
        'boto3',
        'botocore',
        'transformers',
        'diffusers',
        'torch',
        'PIL',
        'matplotlib',
        'wandb'
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(level)

class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding consistent extra information."""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process the logging call to add extra information."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        return msg, kwargs

def create_training_logger(experiment_name: str, 
                         run_id: str,
                         model_name: str = "sdxl") -> logging.Logger:
    """
    Create a specialized logger for training experiments.
    
    Args:
        experiment_name: Name of the experiment
        run_id: Unique run identifier
        model_name: Name of the model being trained
        
    Returns:
        Configured training logger
    """
    config = LogConfig(
        name=f"training.{experiment_name}",
        log_file=f"training_{experiment_name}_{run_id}.log",
        json_file=f"training_{experiment_name}_{run_id}.jsonl",
        enable_metrics=True
    )
    
    logger = setup_logger(config.name, config)
    
    # Create adapter with training-specific context
    adapter = LoggerAdapter(logger, {
        'experiment_name': experiment_name,
        'run_id': run_id,
        'model_name': model_name,
        'component': 'training'
    })
    
    return adapter

def create_inference_logger(model_name: str = "ragamala") -> logging.Logger:
    """
    Create a specialized logger for inference operations.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Configured inference logger
    """
    config = LogConfig(
        name=f"inference.{model_name}",
        log_file=f"inference_{model_name}.log",
        json_file=f"inference_{model_name}.jsonl"
    )
    
    logger = setup_logger(config.name, config)
    
    # Create adapter with inference-specific context
    adapter = LoggerAdapter(logger, {
        'model_name': model_name,
        'component': 'inference'
    })
    
    return adapter

def main():
    """Main function for testing logging utilities."""
    # Test basic logger setup
    logger = setup_logger("test_logger")
    
    # Test basic logging
    logger.info("Testing basic logging functionality")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test contextual logging
    structured_logger = StructuredLogger(LogConfig(name="test_structured"))
    logger = structured_logger.get_logger()
    
    with structured_logger.context(user_id="12345", session_id="abcdef"):
        logger.info("User action performed", extra={
            'action': 'login',
            'ip_address': '192.168.1.1'
        })
    
    # Test performance logging
    perf_logger = PerformanceLogger(logger)
    
    with perf_logger.timer("test_operation"):
        time.sleep(0.1)  # Simulate work
    
    # Test training logger
    training_logger = create_training_logger(
        experiment_name="ragamala_sdxl",
        run_id="run_001",
        model_name="sdxl_lora"
    )
    
    training_logger.info("Training started", extra={
        'epoch': 1,
        'batch_size': 4,
        'learning_rate': 1e-4
    })
    
    # Test inference logger
    inference_logger = create_inference_logger("ragamala_generator")
    
    inference_logger.info("Image generated", extra={
        'prompt': "A rajput style ragamala painting",
        'raga': 'bhairav',
        'style': 'rajput',
        'generation_time': 2.5
    })
    
    print("Logging utilities testing completed!")

if __name__ == "__main__":
    main()

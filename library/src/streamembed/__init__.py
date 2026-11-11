"""StreamEmbed: search your own screen history by describing what you saw. Everything runs on your machine."""
from .api import (Capture, Result, Results, available_index_types, available_models, build_index_from_directory, build_index_from_now, configure, is_paused,
                  pause, query_index, resume, visualize_results)
from .config import get_config, reset_config, set_config
from .service import install_service, service_status, uninstall_service

__version__ = "0.3.2"
__all__ = ["Capture", "Result", "Results", "available_index_types", "available_models", "build_index_from_directory", "build_index_from_now", "configure", "query_index",
           "visualize_results", "install_service", "service_status", "uninstall_service", "pause", "resume", "is_paused",
           "get_config", "set_config", "reset_config"]

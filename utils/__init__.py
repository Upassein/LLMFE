"""LLM-FE utilities package"""

# Re-export functions from utils_legacy.py for backward compatibility
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from the legacy utils module
import utils_legacy
is_categorical = utils_legacy.is_categorical
set_seed = utils_legacy.set_seed  
serialize = utils_legacy.serialize

# Grid template functions
from .grid_template import (
    extract_grid_template,
    apply_template_to_grid,
    merge_all_grids
)

__all__ = [
    'is_categorical',
    'set_seed',
    'serialize',
    'extract_grid_template',
    'apply_template_to_grid',
    'merge_all_grids'
]

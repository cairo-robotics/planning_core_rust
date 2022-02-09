from .cairo_planning_core import *
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))

os.environ["config_path"] = os.path.join(_ROOT, 'data/config/')


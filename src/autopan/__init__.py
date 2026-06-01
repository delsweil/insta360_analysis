# src/autopan — modular autopan components
# These modules can be imported from the root-level scripts by adding
# src/ to sys.path or installing as a package.

from .team_cluster import TeamClusterer, TeamClusterConfig, extract_kit_feature
from .tracking import TrackManager
from .perception import (
    DetBox, Detector, nms_people_xyxy, nms_boxes_xyxy,
    filter_by_y_adaptive_area, pick_best_ball,
)
from .field_opt import FieldViewOptimizer, FieldOptConfig

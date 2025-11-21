"""
Server implementations for FL.
"""

from backfed.servers.base_server import BaseServer
from backfed.servers.defense_categories import (
    ClientSideDefenseServer,
    RobustAggregationServer,
    AnomalyDetectionServer,
    PostAggregationServer,
)
from backfed.servers.trimmed_mean_server import TrimmedMeanServer
from backfed.servers.median_server import GeometricMedianServer, CoordinateMedianServer
from backfed.servers.multi_krum_server import MultiKrumServer, KrumServer, ADMultiKrumServer
from backfed.servers.fedavg_server import UnweightedFedAvgServer, WeightedFedAvgServer
from backfed.servers.fedprox_server import FedProxServer
from backfed.servers.flame_server import FlameServer
from backfed.servers.foolsgold_server import FoolsGoldServer
from backfed.servers.weakdp_server import WeakDPServer, NormClippingServer
from backfed.servers.deepsight_server import DeepSightServer
from backfed.servers.rflbat_server import RFLBATServer
from backfed.servers.fldetector_server import FLDetectorServer
from backfed.servers.fltrust_server import FLTrustServer
from backfed.servers.flare_server import FlareServer
from backfed.servers.robustlr_server import RobustLRServer
from backfed.servers.indicator_server import IndicatorServer
from backfed.servers.localdp_server import LocalDPServer
from backfed.servers.fedavgcka_server import FedAvgCKAServer
from backfed.servers.fedspectre_hybrid_server import FedSPECTREHybridServer
from backfed.servers.fedspectre_stateful_server import FedSPECTREStatefulServer
from backfed.servers.fedspectre_simple_server import FedSPECTRESimpleServer
# Only include FeRA Visualize (exclude other FeRA variants)
from backfed.servers.fera_visualize_server import FeraVisualizeServer

__all__ = [
    # Base classes
    "BaseServer",
    "ClientSideDefenseServer",
    "RobustAggregationServer",
    "AnomalyDetectionServer",
    "PostAggregationServer",

    # Server implementations
    "TrimmedMeanServer",
    "GeometricMedianServer",
    "CoordinateMedianServer",
    "MultiKrumServer",
    "ADMultiKrumServer",
    "KrumServer",
    "UnweightedFedAvgServer",
    "WeightedFedAvgServer",
    "FedProxServer",
    "FlameServer",
    "FoolsGoldServer",
    "NormClippingServer",
    "WeakDPServer",
    "DeepSightServer",
    "RFLBATServer",
    "FLDetectorServer",
    "FLTrustServer",
    "FlareServer",
    "RobustLRServer",
    "IndicatorServer",
    "LocalDPServer",
    "FedAvgCKAServer",
    "FedSPECTREHybridServer",
    "FedSPECTREStatefulServer",
    "FedSPECTRESimpleServer",
    # Only FeRA Visualize included
    "FeraVisualizeServer",
]


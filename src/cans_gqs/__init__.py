"""
CANS/GQS Framework
The Comprehensive Angular Naming System (CANS) and the Geodesic Query System (GQS):
A Formal n-Dimensional Framework for Computational Geometry and Dynamic Simulation

Author: ContributorX Ltd.
"""

__version__ = "1.0.0"
__author__ = "ContributorX Ltd."

from .cans_3d.polyhedral_angle_system import PolyhedralAngleSystem
from .cans_nd.nd_primitives import (
    NDVertex,
    NDEdge,
    NDFace,
    NDHyperface,
    NDPolytope,
    NDGeometryError,
)
from .cans_nd.nd_angular_system import NDAngularSystem
from .gqs.geodesic_query_system import GeodesicQuerySystem

__all__ = [
    "PolyhedralAngleSystem",
    "NDVertex",
    "NDEdge",
    "NDFace",
    "NDHyperface",
    "NDPolytope",
    "NDGeometryError",
    "NDAngularSystem",
    "GeodesicQuerySystem",
]

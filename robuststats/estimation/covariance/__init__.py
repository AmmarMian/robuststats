""" Various robust estimates of second order statistics. 
"""

from .base import (
    RealEmpiricalCovariance,
    RealCovarianceScale,
    ComplexEmpiricalCovariance,
    ComplexCovarianceScale
)

from ._centered_covariance import (
    TylerShapeMatrix,
    ComplexTylerShapeMatrix,
    HuberMEstimator,
    ComplexHuberMEstimator,
    CenteredStudentMLE,
    ComplexCenteredStudentMLE
)

from ._m_estimators import (
    _huber_m_estimator_function,
    _tyler_m_estimator_function,
    fixed_point_m_estimation_centered,
    get_normalisation_function
)
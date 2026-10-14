import pytest
import numpy as np

from shakelab.engineering.structure import (
    building_model,
    F0BuildingModel,
    ElasticParametersBuildingModel,
    VsBuildingModel,
    HalfSpaceModel,
    DamageState,
)
from shakelab.site.response import transfer_function_n_layer
from shakelab.libutils.utils import (
    f0h2vs,
    Q2zeta,
    zeta2Q,
    calculate_stream_difference,
    calc_drift,
)
from shakelab.signals.base import Stream, Record


def test_models_initialization():
    m_f0 = F0BuildingModel(f0=5.0, thickness=10.0, zeta=0.05, rho=2000)
    assert m_f0.vs == 4.0 * 5.0 * 10.0
    assert np.isclose(m_f0.Q, 10.0)

    m_vs = VsBuildingModel(vs=200.0, thickness=10.0, zeta=0.05, rho=2000)
    assert m_vs.vs == 200.0
    assert np.isclose(m_vs.Q, 10.0)

    m_hs = HalfSpaceModel(vs=300.0, zeta=0.05)
    assert m_hs.vs == 300.0
    assert np.isclose(m_hs.Q, 10.0)


def test_transfer_function():
    building = F0BuildingModel(f0=5.0, thickness=10.0, zeta=0.05, rho=2000)
    half_space = HalfSpaceModel(vs=300.0, zeta=0.05)
    freqs = np.linspace(0.1, 20.0, 50)
    tf = transfer_function_n_layer(freqs, building, half_space)
    assert len(tf) == 50
    assert np.all(np.isfinite(tf))


def test_damage_state():
    ds = DamageState(
        method="scaini_etal",
        drift=0.5,
        construction_material="URM",
        construction_year=1940,
        number_of_floor=2,
    )
    assert ds.get_damage_stage() == "Slight or None"

    ds_ext = DamageState(
        method="scaini_etal",
        drift=1.0,
        construction_material="URM",
        construction_year=1940,
        number_of_floor=2,
    )
    assert ds_ext.get_damage_stage() == "Extensive"


def test_stream_difference_and_drift():
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([0.5, 1.0, 1.5])
    rec1 = Record(data=data1, delta=0.01)
    rec2 = Record(data=data2, delta=0.01)
    st1 = Stream()
    st2 = Stream()
    st1.record.append(rec1)
    st2.record.append(rec2)

    st_diff = calculate_stream_difference(st1, st2)
    assert np.allclose(st_diff.record[0].data, [0.5, 1.0, 1.5])

    drift = calc_drift(st1, st2)
    assert np.isclose(drift, 1.5)

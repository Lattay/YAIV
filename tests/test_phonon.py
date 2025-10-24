from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose

from yaiv import grep, utils as ut
from yaiv.defaults.config import ureg
from yaiv import phonon as ph

RESULTS_DIRS = [
    "qe/results",
]
IDS = RESULTS_DIRS[:]


def test_QEdyn2Realdyn_synthetic():
    m = np.array([1.0, 4.0]) * ureg._2m_e
    K11 = np.eye(3) * 2.0
    K22 = np.eye(3) * 3.0
    K12 = np.ones((3, 3)) * 0.5
    K21 = K12.T

    K_phys = np.zeros((6, 6))
    K_phys[:3, :3] = K11
    K_phys[3:, 3:] = K22
    K_phys[:3, 3:] = K12
    K_phys[3:, :3] = K21

    M = np.zeros_like(K_phys)
    for i in range(2):
        for j in range(2):
            si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            M[si, sj] = K_phys[si, sj] * np.sqrt(m[i].magnitude * m[j].magnitude)

    Mq = M * (ureg("_2m_e*Ry^2/planck_constant^2"))
    R = ph._QEdyn2Realdyn(Mq, m)
    assert isinstance(R, ureg.Quantity)
    assert R.check(ureg("Ry^2/planck_constant^2"))
    assert_allclose(R.magnitude, K_phys, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rel_dir", RESULTS_DIRS, ids=IDS)
def test_dyn_from_file_and_diagonalize(data_dir, require, rel_dir):
    results_dir = data_dir / rel_dir
    require(results_dir, f"Missing results directory: {rel_dir}")
    if not list(results_dir.glob("*.dyn*")):
        pytest.skip(f"No .dyn* files in {results_dir}")

    q = np.array([0, 0, 0])
    d = ph.Dyn.from_file(q_cryst=q, results_ph_path=str(results_dir), qe_format=True)

    assert d.q.check(ureg._2pi / ureg.crystal)
    assert hasattr(d, "dyn") and hasattr(d, "masses") and hasattr(d, "Cell")
    assert d.masses.check(ureg._2m_e)

    d.diagonalize(qe_format_in=True, qe_format_out=True)
    assert hasattr(d, "freqs")
    assert d.freqs.check(ureg.c / ureg.cm)
    assert d.freqs.magnitude.ndim == 1
    assert hasattr(d, "displacements") and hasattr(d, "polarizations")
    n_atoms = len(d.Cell[1])
    assert d.displacements.shape[1:] == (n_atoms, 3)
    assert d.polarizations.shape[1:] == (n_atoms, 3)


def test_find_supercell_simple_cases():
    out = ph._find_supercell(np.array([0.5, 0.0, 0.0]))
    assert np.array_equal(out.size, np.array([2, 1, 1]))
    pf = out.phase_factors[0]
    assert pf.shape == tuple(out.size)
    assert_allclose(pf[0, 0, 0], 1.0 + 0.0j)
    assert_allclose(pf[1, 0, 0], np.exp(2j * np.pi * 0.5))

    out2 = ph._find_supercell(
        [np.array([0.5, 0.0, 0.0]), np.array([0.0, 1.0 / 3.0, 0.0])]
    )
    assert np.array_equal(out2.size, np.array([2, 3, 1]))
    assert len(out2.phase_factors) == 2
    for q in out2.q:
        assert q.check(ureg._2pi / ureg.crystal)


@pytest.mark.parametrize("rel_dir", RESULTS_DIRS, ids=IDS)
def test_cdw_from_file_and_distort(data_dir, require, rel_dir):
    results_dir = data_dir / rel_dir
    require(results_dir, f"Missing results directory: {rel_dir}")
    q = np.array([0, 0, 0])
    try:
        cdw = ph.CDW.from_files(q_cryst=q, results_ph_path=str(results_dir))
    except FileNotFoundError:
        pytest.skip(f"q={q} not found in {results_dir}")

    assert hasattr(cdw, "SuperCell")
    assert hasattr(cdw, "q")
    assert cdw.masses.check(ureg._2m_e)

    amp = np.array([0.05]) * ureg.angstrom
    distorted = cdw.distort_crystal(amplitudes=amp, modes=[0])
    assert len(distorted.spglib[2]) == len(cdw.SuperCell.spglib[2])
    disp = distorted.atoms.positions - cdw.SuperCell.atoms.positions
    assert np.linalg.norm(disp) > 0.0


@pytest.mark.parametrize("rel_dir", RESULTS_DIRS, ids=IDS)
def test_boes_save_load_and_line_generation(data_dir, require, tmp_path, rel_dir):
    results_dir = data_dir / rel_dir
    require(results_dir, f"Missing results directory: {rel_dir}")
    q = np.array([0, 0, 0])
    try:
        cdw = ph.CDW.from_files(q_cryst=q, results_ph_path=str(results_dir))
    except FileNotFoundError:
        pytest.skip(f"q={q} not found in {results_dir}")

    boes = ph.BOES(cdw)

    amp_i = np.array([-0.05]) * ureg.angstrom
    amp_f = np.array([0.05]) * ureg.angstrom
    boes.generate_structures_line(
        amplitude_i=amp_i, amplitude_f=amp_f, modes=[0], steps=3
    )

    assert boes.structures is not None
    assert boes.amplitudes is not None
    assert boes.order_parameters is not None
    assert len(boes.structures) == 3

    out = tmp_path / "boes.pkl"
    boes.save_as(str(out))
    assert out.exists()
    boes2 = ph.BOES.from_file(str(out))
    assert isinstance(boes2, ph.BOES)
    assert len(boes2.structures) == len(boes.structures)

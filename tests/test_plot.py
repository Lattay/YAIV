import numpy as np
import matplotlib.pyplot as plt
import pytest
from types import SimpleNamespace

from yaiv import plot as plot
from yaiv.spectrum import ElectronBands, PhononBands
from yaiv import grep
from yaiv.defaults.config import ureg


# Define explicit file combinations (relative to tests/data)
# Provide one path per source; if optional entries (kpath/lattice) are provided,
# they will be required and used; otherwise they are skipped.
BANDS_CASES = [
    SimpleNamespace(
        name="vasp_outcar_kpath_poscar",
        bands="vasp/RESULTS/OUTCAR_BS",  # required
        kpath="vasp/KPATH",  # optional but provided here
        lattice="vasp/POSCAR",  # optional but provided here
        window_eV=5,
    ),
    SimpleNamespace(
        name="vasp_eigenval_kpath_poscar",
        bands="vasp/RESULTS/EIGENVAL_BS",
        kpath="vasp/KPATH",
        lattice="vasp/POSCAR",
        window_eV=5,
    ),
    SimpleNamespace(
        name="qe_xml_bands_in",
        bands="qe/results/Si.xml",
        kpath="qe/results/Si.bands.pwi",
        window_eV=5,
    ),
]

PHONON_CASES = [
    SimpleNamespace(
        name="qe_freq_matdyn",
        phonon="qe/results/Si.freq",  # required
        kpath="qe/results/matdyn.in",  # optional but provided here (for labels)
        lattice="qe/results/Si.scf.pwo",
    ),
]


def build_electron_bands(case, data_dir, require):
    # Required bands file
    p_bands = data_dir / case.bands
    require(p_bands, f"Missing test data: {case.bands}")
    bands = ElectronBands(str(p_bands))

    # Optional kpath override
    if getattr(case, "kpath", None):
        p_kpath = data_dir / case.kpath
        require(p_kpath, f"Missing kpath data: {case.kpath}")
        bands.kpath = grep.kpath(str(p_kpath), labels=True)

    # Optional lattice override
    if getattr(case, "lattice", None):
        p_lat = data_dir / case.lattice
        require(p_lat, f"Missing lattice data: {case.lattice}")
        L = grep.lattice(str(p_lat))
        bands.lattice = L  # Spectrum.lattice setter updates k_lattice

    return bands


def build_phonon_bands(case, data_dir, require):
    p_ph = data_dir / case.phonon
    require(p_ph, f"Missing test data: {case.phonon}")
    phonons = PhononBands(str(p_ph))

    if getattr(case, "kpath", None):
        p_kpath = data_dir / case.kpath
        require(p_kpath, f"Missing kpath data: {case.kpath}")
        kp = grep.kpath(str(p_kpath), labels=True)
        phonons.kpath = kp

    # Optional lattice override
    if getattr(case, "lattice", None):
        p_lat = data_dir / case.lattice
        require(p_lat, f"Missing lattice data: {case.lattice}")
        L = grep.lattice(str(p_lat))
        phonons.lattice = L  # Spectrum.lattice setter updates k_lattice
    return phonons


tolerance = 5


# ---------- Bands only ----------
@pytest.mark.parametrize("case", BANDS_CASES, ids=[c.name for c in BANDS_CASES])
@pytest.mark.mpl_image_compare(
    style="default", tolerance=tolerance, savefig_kwargs={"bbox_inches": "tight"}
)
def test_bands_image_and_structure_cases(data_dir, require, case):
    bands = build_electron_bands(case, data_dir, require)
    fig, ax = plt.subplots()
    win = getattr(case, "window_eV", 1.5) * ureg.eV
    plot.bands(bands, ax=ax, window=win)

    # Structural checks
    assert len(ax.get_lines()) >= 1
    x0, x1 = ax.get_xlim()
    assert x0 == pytest.approx(0.0) and x1 == pytest.approx(1.0)
    y0, y1 = ax.get_ylim()
    assert y1 - y0 > 0
    if bands.fermi is not None:
        found_zero = any(
            (ln.get_ydata()[0] == 0 and ln.get_ydata()[-1] == 0) for ln in ax.lines
        )
        assert found_zero

    ax.set_title(f"Bands ({case.name})")
    fig.tight_layout()
    return fig


# ---------- DOS only ----------
@pytest.mark.parametrize("case", BANDS_CASES, ids=[c.name for c in BANDS_CASES])
@pytest.mark.mpl_image_compare(
    style="default", tolerance=tolerance, savefig_kwargs={"bbox_inches": "tight"}
)
def test_dos_image_and_structure_cases(data_dir, require, case):
    bands = build_electron_bands(case, data_dir, require)

    fig, ax = plt.subplots()
    plot.DOS(
        bands,
        ax=ax,
        window=1.0 * ureg.eV,
        order=0,
        cutoff_sigmas=3.0,
        fill=True,
        alpha=0.4,
    )

    # Structural checks
    assert len(ax.lines) >= 1
    x0, x1 = ax.get_xlim()
    assert x1 - x0 > 0
    if bands.fermi is not None:
        found_zero = any(
            (ln.get_xdata()[0] == 0 and ln.get_xdata()[-1] == 0) for ln in ax.lines
        )
        assert found_zero

    ax.set_title(f"DOS ({case.name})")
    fig.tight_layout()
    return fig


# ---------- Bands + DOS panel ----------
@pytest.mark.parametrize("case", BANDS_CASES, ids=[c.name for c in BANDS_CASES])
@pytest.mark.mpl_image_compare(
    style="default", tolerance=tolerance, savefig_kwargs={"bbox_inches": "tight"}
)
def test_bands_dos_panel_image_and_structure_cases(data_dir, require, case):
    bands = build_electron_bands(case, data_dir, require)

    fig = plt.figure(figsize=(6, 3))
    ax_b, ax_d = plot.bandsDOS(
        bands, fig=fig, window=getattr(case, "window_eV", 1.5) * ureg.eV
    )

    # Bands axis structural checks
    assert len(ax_b.get_lines()) >= 1
    x0, x1 = ax_b.get_xlim()
    assert x0 == pytest.approx(0.0) and x1 == pytest.approx(1.0)
    # DOS axis label
    assert ax_d.get_xlabel() == "DOS"

    fig.suptitle(f"Bands + DOS ({case.name})")
    fig.tight_layout()
    return fig


# ---------- Phonon: bands and panel ----------
@pytest.mark.parametrize("case", PHONON_CASES, ids=[c.name for c in PHONON_CASES])
@pytest.mark.mpl_image_compare(
    style="default", tolerance=tolerance, savefig_kwargs={"bbox_inches": "tight"}
)
def test_phonon_bands_image_and_structure_cases(data_dir, require, case):
    phonons = build_phonon_bands(case, data_dir, require)

    fig, ax = plt.subplots()
    plot.phonons(phonons, ax=ax, window=None)

    assert len(ax.get_lines()) >= 1
    x0, x1 = ax.get_xlim()
    assert x0 == pytest.approx(0.0) and x1 == pytest.approx(1.0)
    # y=0 line present
    found_zero = any(
        (ln.get_ydata()[0] == 0 and ln.get_ydata()[-1] == 0) for ln in ax.lines
    )
    assert found_zero

    ax.set_title(f"Phonon bands ({case.name})")
    fig.tight_layout()
    return fig


@pytest.mark.parametrize("case", PHONON_CASES, ids=[c.name for c in PHONON_CASES])
@pytest.mark.mpl_image_compare(
    style="default", tolerance=tolerance, savefig_kwargs={"bbox_inches": "tight"}
)
def test_phonon_dos_panel_image_and_structure_cases(data_dir, require, case):
    phonons = build_phonon_bands(case, data_dir, require)

    fig = plt.figure(figsize=(6, 3))
    ax_b, ax_d = plot.phononsDOS(phonons, fig=fig, window=None)

    assert len(ax_b.get_lines()) >= 1
    x0, x1 = ax_b.get_xlim()
    assert x0 == pytest.approx(0.0) and x1 == pytest.approx(1.0)
    assert ax_d.get_xlabel() == "DOS"

    fig.suptitle(f"Phonons + DOS ({case.name})")
    fig.tight_layout()
    return fig

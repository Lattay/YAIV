import pytest

from yaiv import convergence as conv

cutoff_file = "others/convergence_cutoff.pkl"
kgrid_file = "others/convergence_kgrid.pkl"


tolerance = 5


@pytest.mark.mpl_image_compare(
    style="default", tolerance=tolerance, savefig_kwargs={"bbox_inches": "tight"}
)
def test_cutoff_analysis(data_dir, require):
    f = data_dir / cutoff_file
    require(f, f"Missing test data: {cutoff_file}")
    C = conv.Self_consistent.from_pkl(f)
    fig, ax = C.analyze.cutoff()
    return fig


@pytest.mark.mpl_image_compare(
    style="default", tolerance=tolerance, savefig_kwargs={"bbox_inches": "tight"}
)
def test_kgrid_analysis(data_dir, require):
    f = data_dir / kgrid_file
    require(f, f"Missing test data: {kgrid_file}")
    C = conv.Self_consistent.from_pkl(f)
    fig, ax = C.analyze.kgrid()
    return fig

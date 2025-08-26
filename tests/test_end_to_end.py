import pandas as pd
import pytest

from cgr_smiles.cgr_to_rxn import cgrsmiles_to_rxnsmiles
from cgr_smiles.rxn_to_cgr import rxnsmiles_to_cgrsmiles
from cgr_smiles.utils import ROOT_DIR, canonicalize

TEST_DATA_PATH = ROOT_DIR / "data" / "cgr"


@pytest.fixture(scope="session")
def individual_test_cases():
    """Load and prepare individual reaction test cases for the entire test session.

    Reads multiple CSV datasets and returns a list of test cases and corresponding IDs.
    Each test case is a tuple containing:
        - file_path (str): Path to the CSV file.
        - idx (int): Row index in the CSV.
        - rxn (str): Reaction SMILES string.
        - rxn_col (str): Column name containing the reaction SMILES.

    Returns:
        tuple[list[tuple[str, int, str, str]], list[str]]:
            A tuple containing the list of test cases and a list of test IDs.
    """
    test_cases = []
    ids = []

    dataset_configs = [
        ("rgd1/rgd1_full.csv", "smiles"),
        ("e2/test.csv", "AAM"),
        ("sn2/test.csv", "AAM"),
        ("rdb7/test.csv", "smiles"),
        ("rdb7/val.csv", "smiles"),
        ("rdb7/train.csv", "smiles"),
        ("cycloaddition/full_dataset.csv", "rxn_smiles"),
    ]

    for file_path, rxn_col in dataset_configs:
        full_path = TEST_DATA_PATH / file_path
        df = pd.read_csv(full_path)

        for idx, row in df.iterrows():
            rxn = row[rxn_col]
            test_cases.append((file_path, idx, rxn, rxn_col))
            ids.append(f"{file_path}:{idx}")

    return test_cases, ids


@pytest.mark.parametrize(
    "file_path, idx, rxn_smiles, rxn_col",
    *individual_test_cases(),  # unpack fixture return value
)
def test_roundtrip_per_sample(file_path, idx, rxn_smiles, rxn_col):
    """Test single sample roundtrip (RXN -> CGR -> RXN)."""
    rxn_can = canonicalize(rxn_smiles)
    cgr = rxnsmiles_to_cgrsmiles(rxn_smiles, keep_atom_mapping=True)
    res = cgrsmiles_to_rxnsmiles(cgr)
    res_can = canonicalize(res)

    assert (
        res_can == rxn_can
    ), f"Mismatch at {file_path}:{idx}, cgr={cgr}, rxn_can={rxn_can}, res_can={res_can}"

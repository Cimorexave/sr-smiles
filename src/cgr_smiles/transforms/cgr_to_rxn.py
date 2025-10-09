import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem

from cgr_smiles.logger import logger
from cgr_smiles.transforms.rxn_to_cgr import remove_redundant_brackets
from cgr_smiles.utils import (
    ORGANIC_SUBSET,
    TokenType,
    _tokenize,
    common_elements_preserving_order,
    extract_chiral_tag_by_atom_map_num,
    flip_e_z_stereo,
    get_atom_map_adjacency_list_from_smiles,
    is_num_permutations_even,
    remove_atom_mapping,
)


def parse_bonds_from_smiles(smiles: str) -> Dict[Tuple[int, int], str]:
    """Parses SMILES to map bond atom-map pairs to their bond specifiers.

    This function traverses the SMILES token by token, identifying bonds by
    their connecting atom map numbers and extracting their explicit bond type.

    Args:
        smiles (str): SMILES string of a molecule.

    Returns:
        Dict[Tuple[int, int], str]: A dictionary mapping sorted `(atom_map_num_1, atom_map_num_2)`
            tuples to their bond specifier string.

    Raises:
        ValueError: If the CGR SMILES string has malformed syntax.
    """
    replace_dict_bonds = {}
    anchor_logical_idx = None
    next_bond_specifier = None
    branches = []
    ring_open_bonds = {}

    logical_idx_to_map_num = {}
    current_logical_idx = 0

    for tokentype, token_original_idx, token_val in _tokenize(smiles):
        if tokentype == TokenType.ATOM:
            # extract atom map number or assign a temporary one if none.
            atom_map_match = re.search(r":(\d+)", str(token_val))
            current_atom_map_num = (
                int(atom_map_match.group(1)) if atom_map_match else (current_logical_idx + 1000)
            )

            logical_idx_to_map_num[current_logical_idx] = current_atom_map_num

            if anchor_logical_idx is not None:
                # we have a bond between anchor_logical_idx and current_logical_idx
                bond_map_num_pair = (
                    logical_idx_to_map_num[anchor_logical_idx],
                    current_atom_map_num,
                )

                # Determine the bond specification
                # If next_bond_specifier is None, it implies a single bond by default.
                bond_val = next_bond_specifier if next_bond_specifier is not None else "-"
                replace_dict_bonds[bond_map_num_pair] = bond_val

            anchor_logical_idx = current_logical_idx
            current_logical_idx += 1
            next_bond_specifier = None  # Clear any pending bond specifier

        elif tokentype == TokenType.BOND_TYPE or tokentype == TokenType.EZSTEREO:
            # These are standard bond types (-, =, #, : or E/Z stereo)
            next_bond_specifier = str(token_val)

        elif tokentype == TokenType.BRANCH_START:
            branches.append(anchor_logical_idx)

        elif tokentype == TokenType.BRANCH_END:
            if not branches:
                raise ValueError(f"Unmatched ')' in SMILES string at index {token_original_idx}")
            anchor_logical_idx = branches.pop()
            next_bond_specifier = None

        elif tokentype == TokenType.RING_NUM:
            ring_num_val = str(token_val)

            if ring_num_val in ring_open_bonds:  # found a matching ring closer
                logical_idx_opener, bond_opener_specifier = ring_open_bonds[ring_num_val]

                # Bond is between the current atom (anchor_logical_idx) and the atom that opened the ring
                bond_map_num_pair = (
                    logical_idx_to_map_num[anchor_logical_idx],
                    logical_idx_to_map_num[logical_idx_opener],
                )

                # Determine the bond specification for this ring closure
                # If there's a bond specifier immediately before this ring_num_val, use it.
                # Otherwise, use the bond specifier that was active when the ring *opened*.
                bond_val = (
                    next_bond_specifier
                    if next_bond_specifier is not None
                    else (bond_opener_specifier if bond_opener_specifier is not None else "-")
                )

                replace_dict_bonds[bond_map_num_pair] = bond_val

                del ring_open_bonds[ring_num_val]  # Remove from open rings
                next_bond_specifier = None  # Clear any pending bond specifier

            else:
                ring_open_bonds[ring_num_val] = (
                    anchor_logical_idx,
                    next_bond_specifier,
                )
                next_bond_specifier = None

    return replace_dict_bonds


def remove_bonds_by_atom_map_nums(mol: Chem.Mol, atom_map_pairs: List[Tuple[int, int]]) -> Chem.Mol:
    """Removes specified bonds from an RDKit molecule based on atom map number pairs.

    Args:
        mol (Chem.Mol): The input RDKit molecule object.
        atom_map_pairs (List[Tuple[int, int]]): A list of atom map tuples to be removed.

    Returns:
        Chem.Mol: A new RDKit molecule object with the specified bonds removed.
            If a bond corresponding to a given pair of atom map numbers does not exist,
            a warning is logged, and that pair is skipped.

    """
    atom_map_to_idx = {}
    for atom in mol.GetAtoms():
        atom_map_to_idx[atom.GetAtomMapNum()] = atom.GetIdx()

    emol = Chem.EditableMol(mol)

    bonds_to_remove_by_idx = []
    for am1, am2 in atom_map_pairs:
        idx1 = atom_map_to_idx[am1]
        idx2 = atom_map_to_idx[am2]

        bond = mol.GetBondBetweenAtoms(idx1, idx2)
        if bond:
            bonds_to_remove_by_idx.append((idx1, idx2))
        else:
            logger.warning(f"No bond found between atom map numbers {am1} and {am2}. Skipping removal.")

    for idx1, idx2 in bonds_to_remove_by_idx:
        emol.RemoveBond(idx1, idx2)

    final_mol = emol.GetMol()
    return final_mol


def update_chirality_tags(smiles: str, cgr_scaffold: str, chiral_center_map_nums: List[int]) -> str:
    """Updates chirality tags in a SMILES string based on a CGR scaffold.

    Identifies chiral centers in the provided RDKit molecule (`mol`) by their atom
    map numbers. It then compares the neighborhood of these chiral centers in
    both the input SMILES (`smiles`) and a reference CGR scaffold (`cgr_scaffold`)
    to determine the chirality tags (@ or @@). If the chirality appears inverted
    between the SMILES and scaffold, the tag is flipped.

    Args:
        smiles (str): The input SMILES string of the molecule.
        cgr_scaffold (List[int]): A reference CGR SMILES string containing correct chirality
            information for comparison.
        chiral_center_map_nums: List of the atom map numbers of the chiral centers.

    Returns:
        A new SMILES string with updated or corrected chirality tags.

    """
    reac_adj = get_atom_map_adjacency_list_from_smiles(smiles)
    cgr_adj = get_atom_map_adjacency_list_from_smiles(cgr_scaffold)

    reac_tokens = [[tok_type, tok] for tok_type, _, tok in _tokenize(smiles)]
    for i, (tok_type, tok) in enumerate(reac_tokens):
        if tok_type == TokenType.ATOM:
            match = re.search(r":(\d+)", tok)
            map_num = int(match.group(1))
            if map_num in chiral_center_map_nums:
                reac_nbrs = reac_adj[map_num]
                cgr_nbrs = cgr_adj[map_num]
                reac_nbrs, cgr_nbrs = common_elements_preserving_order(reac_nbrs, cgr_nbrs)

                current_tag = extract_chiral_tag_by_atom_map_num(cgr_scaffold, map_num)

                if is_num_permutations_even(reac_nbrs, cgr_nbrs):
                    chirality_tag = current_tag
                else:
                    if current_tag == "@":
                        chirality_tag = "@@"
                    elif current_tag == "@@":
                        chirality_tag = "@"

                # replace_pattern = rf"(\[[A-Z][a-z]?)(@{{1,2}})?(:{map_num}\])"
                replace_pattern = rf"(\[[A-Z][a-z]?)(@{{1,2}})?([+-]*:{map_num}\])"
                old_tok = reac_tokens[i][1]
                tok = re.sub(replace_pattern, rf"\1{chirality_tag}\3", old_tok)
                reac_tokens[i][1] = tok

    return "".join([str(tok[1]) for tok in reac_tokens])


def find_cis_trans_stereo_bonds(
    bond_dict: Dict[Tuple[int, int], str],
) -> Dict[Tuple[int, int], Dict[str, any]]:
    r"""Identifies cis/trans stereochemistry of double bonds from bond data.

    Parses a dictionary representing molecule bonds and their types. It identifies
    double bonds and determines their cis/trans stereochemistry by examining
    stereo bond specifiers ('/' or '\') on adjacent bonds connected to the
    double bond atoms.

    Args:
        bond_dict (Dict[Tuple[int, int], str]): A dictionary where keys are tuples of atom indices,
            and values are strings indicating the bond type.

    Returns:
        Dict[Tuple[int, int], Dict[str, any]: A dictionary where keys are tuples of atom indices
            `(idx_a, idx_b)` representing the double bond, and values are dictionaries containing:
                - "stereo": An RDKit `Chem.BondStereo` enum value (STEREOCIS or STEREOTRANS).
                - "terminal_atoms": A tuple of the two atom indices `(neighbor_a_idx, neighbor_b_idx)`
                    that define the stereochemistry for each side of the double bond.
                    The dictionary contains entries for both `(a, b)` and `(b, a)` for the same
                    double bond.
    """
    if not bond_dict:
        return {}

    neighbors_map = defaultdict(list)
    stereo_bond_map = {}
    double_bonds = set()

    for (a1, a2), bond_type in bond_dict.items():
        neighbors_map[a1].append((a2, bond_type))
        neighbors_map[a2].append((a1, flip_e_z_stereo(bond_type)))

        if bond_type == "=":
            double_bonds.add(frozenset((a1, a2)))
        elif bond_type in ("/", "\\"):
            stereo_bond_map[(a1, a2)] = bond_type
            stereo_bond_map[(a2, a1)] = flip_e_z_stereo(bond_type)

    results = {}

    for db_pair in double_bonds:
        atom_a, atom_b = tuple(db_pair)

        stereo_arms_a = []
        for neighbor_atom_a, _ in neighbors_map.get(atom_a, []):
            if (atom_a, neighbor_atom_a) in stereo_bond_map:
                stereo_arms_a.append(
                    (
                        atom_a,
                        neighbor_atom_a,
                        stereo_bond_map[(atom_a, neighbor_atom_a)],
                    )
                )

        stereo_arms_b = []
        for neighbor_atom_b, _ in neighbors_map.get(atom_b, []):
            if (atom_b, neighbor_atom_b) in stereo_bond_map:
                stereo_arms_b.append(
                    (
                        atom_b,
                        neighbor_atom_b,
                        stereo_bond_map[(atom_b, neighbor_atom_b)],
                    )
                )

        if stereo_arms_a and stereo_arms_b:
            arm_a = stereo_arms_a[0]
            arm_b = stereo_arms_b[0]

            neighbor_a, slash_type_a = arm_a[1], flip_e_z_stereo(arm_a[2])
            neighbor_b, slash_type_b = arm_b[1], arm_b[2]

            if slash_type_a == slash_type_b:
                stereo = Chem.BondStereo.STEREOE
            else:
                stereo = Chem.BondStereo.STEREOZ

            results[(atom_a, atom_b)] = {
                "stereo": stereo,
                "terminal_atoms": (neighbor_a, neighbor_b),
            }
            results[(atom_b, atom_a)] = {
                "stereo": stereo,
                "terminal_atoms": (neighbor_b, neighbor_a),
            }

    return results


def update_cis_trans_stereo_chem(mol: Chem.Mol, parsed_bonds: dict) -> Chem.Mol:
    """Update cis/trans stereochemistry for double bonds in a molecule.

    This function uses pre-parsed bond and stereochemistry information to correct
    the cis/trans configuration of double bonds in an RDKit molecule. Atom map numbers
    are preserved, and stereochemistry is updated according to the provided bond data.

    Args:
        mol (Chem.Mol): An RDKit molecule object with atom map numbers.
        parsed_bonds (dict): A dictionary where keys are bond identifiers (tuple of atom map numbers),
            and values are dictionaries containing:
                - 'terminal_atoms' (Tuple[int, int]): The atom map numbers of the bond ends.
                - 'stereo' (Chem.rdchem.BondStereo): The desired stereochemistry for the bond.

    Returns:
        Chem.Mol: The input molecule with updated cis/trans stereochemistry on relevant bonds.
    """
    b = find_cis_trans_stereo_bonds(parsed_bonds)

    # assigning stereochem manually only works for individual molecules, i.e. smiles
    # does not include ".". Therefore, iterate over the fragments of the molecule.
    frags = Chem.GetMolFrags(mol, asMols=True)

    for frag in frags:
        map_num2idx = {a.GetAtomMapNum(): a.GetIdx() for a in frag.GetAtoms()}
        for bond in frag.GetBonds():
            am1 = bond.GetBeginAtom().GetAtomMapNum()
            am2 = bond.GetEndAtom().GetAtomMapNum()

            if (am1, am2) in b.keys():
                nbr1, nbr2 = b[(am1, am2)]["terminal_atoms"]
                nbr1, nbr2 = map_num2idx[nbr1], map_num2idx[nbr2]
                bond.SetStereoAtoms(nbr1, nbr2)

                stereo = b[(am1, am2)]["stereo"]
                bond.SetStereo(stereo)

            elif (am2, am1) in b.keys():
                nbr1, nbr2 = b[(am2, am1)]["terminal_atoms"]
                nbr1, nbr2 = map_num2idx[nbr1], map_num2idx[nbr2]
                bond.SetStereoAtoms(nbr2, nbr1)

                stereo = b[(am2, am1)]["stereo"]
                bond.SetStereo(stereo)

        Chem.AssignStereochemistry(frag, force=True)

    smiles = [Chem.MolToSmiles(f, canonical=False) for f in frags]
    m = Chem.MolFromSmiles(".".join(smiles), sanitize=False)
    return m


def get_reac_prod_scaffold_smiles_from_cgr(cgr_smiles: str) -> Tuple[str, str]:
    """Extracts the reactant and product scaffold SMILES from a CGR SMILES string.

    The CGR SMILES encodes atom-level differences between reactants and products using
    substitution patterns in the form `{reactant|product}`.
    This function decodes those patterns by replacing each `{...|...}` block with the
    appropriate fragment in two parallel SMILES strings: one for the reactant, one for the product.

    Args:
        cgr_smiles (str): A CGR SMILES string containing substitution patterns.

    Returns:
        Tuple[str, str]: A tuple containing the reactant SMILES and product SMILES
            with all substitution patterns resolved.
    """
    reac_smi = cgr_smiles
    prod_smi = cgr_smiles

    cgr_pattern = r"\{([^{|}]*)\|([^{|}]*)\}"

    while "{" in reac_smi:
        match = re.search(cgr_pattern, reac_smi)
        if match is None:
            break

        full_match = match.group(0)
        reac_fragment = match.group(1)
        prod_fragment = match.group(2)

        # replace the first match occurrence
        reac_smi = reac_smi.replace(full_match, reac_fragment, 1)
        prod_smi = prod_smi.replace(full_match, prod_fragment, 1)

    return reac_smi, prod_smi


def get_chiral_center_map_nums(mol: Chem.Mol) -> List[int]:
    """Returns the atom map numbers of chiral centers in an RDKit molecule.

    Args:
        mol (Chem.Mol): The input RDKit molecule object.

    Returns:
        List[int]: A list of integer atom map numbers corresponding to the chiral centers
            found in the molecule.
    """
    atom_map_nums = []
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        ):
            atom_map_nums.append(atom.GetAtomMapNum())
    return atom_map_nums


class CgrToRxn:
    """Transform reaction SMILES into CGR SMILES.

    This class provides a callable interface to convert CGR SMILES into reaction
    SMILES. It supports single strings, lists of strings, pandas Series, and
    pandas DataFrames.

    Attributes:
        cgr_col (Optional[str]): Column name in a DataFrame containing CGR SMILES.
        add_atom_mapping (bool, optional): If True, ensures atom mappings are
            present in the output RXN SMILES. If False, atom mappings are stripped
            unless they were already present in the input. Default is False.

    Examples:
        Transform a pandas DataFrame of reactions into CGR SMILES:

        >>> import pandas as pd
        >>> df = pd.read_csv("path/to/file.csv")
        >>> transform = CgrToRxn(cgr_col="cgr_smiles")
        >>> df["rxn_smiles"] = transform(df)
    """

    def __init__(
        self,
        cgr_col: Optional[str] = None,
        add_atom_mapping: bool = False,
    ) -> None:
        """Initialize the transformation object.

        Args:
            cgr_col (str, optional): Column name in a DataFrame containing
                CGR SMILES. Required if passing a DataFrame. Defaults to None.
            add_atom_mapping (bool, optional): If True, ensures atom mappings are
                present in the output RXN SMILES. If False, atom mappings are stripped
                unless they were already present in the input. Default is False.

        """
        self.cgr_col = cgr_col
        self.add_atom_mapping = add_atom_mapping

    def __call__(
        self, data: Union[str, List[str], pd.Series, pd.DataFrame]
    ) -> Union[str, List[str], pd.Series, pd.DataFrame]:
        """Apply the transformation to CGR SMILES.

        Args:
            data (Union[str, List[str], pd.Series, pd.DataFrame]): Input data
                containing CGR SMILES. Can be a single string, a list of strings,
                a pandas Series, or a pandas DataFrame.

        Returns:
            Union[str, List[str], pd.Series, pd.DataFrame]: Transformed CGR SMILES
            in the same structure as the input.

        Raises:
            ValueError: If a DataFrame is provided but `self.cgr_col` is not set.
            TypeError: If the input type is not supported.
        """
        if isinstance(data, str):
            return cgr_to_rxn(data, self.add_atom_mapping)

        elif isinstance(data, list):
            return [self(d) for d in data]

        elif isinstance(data, pd.Series):
            return data.apply(self)

        elif isinstance(data, pd.DataFrame):
            if self.cgr_col is None:
                raise ValueError(
                    f"A pandas DataFrame was provided, but `self.cgr_col` is not set.\n"
                    f"Available columns are: {list(data.columns)}\n"
                    "Please specify the column name containing the reactions by setting "
                    "`cgr_col` at time of initialization."
                )
            return data[self.cgr_col].apply(self)

        else:
            raise TypeError("Input must be str, list, pandas Series, or DataFrame.")


def is_cgr_smiles_fully_atom_mapped(cgr_smiles: str) -> bool:
    """Checks if a CGR SMILES string is fully atom-mapped.

    Checks according to the following definition:
    - All CGRTOKENs ({...|...}) must have both alternatives atom-mapped with
      the SAME map number.
    - All TOKENs ([...]) must be atom-mapped.
    """
    atom_map_pattern = re.compile(r":\d+")

    for tok_type, _, tok in _tokenize(cgr_smiles):
        if tok_type == TokenType.ATOM:
            if not atom_map_pattern.search(tok):
                return False

    return True


def add_atom_mapping_to_cgr(cgr: str) -> str:
    """Add atom mapping numbers to a CGR-SMILES string.

    Each atom gets a continuous unique index: 1, 2, 3, ...
    Atoms inside the same {...|...} group share one index.
    """
    atom_pattern = re.compile(r"(\[[^\]]+\]|[A-Z][a-z]?|[cnops])")
    mapping_counter = 1

    def insert_mapping(atom, idx):
        if atom.startswith("["):
            if re.search(r":\d+", atom) is not None:  # already has mapping
                return atom
            return atom[:-1] + f":{idx}]"
        else:
            return f"[{atom}:{idx}]"

    out = []
    i = 0

    while i < len(cgr):
        if cgr[i] == "{":  # handle group {...|...}
            j = cgr.find("}", i)
            group_content = cgr[i + 1 : j]

            # recursively map inside group using same index
            group_mapped = atom_pattern.sub(
                lambda m: insert_mapping(m.group(), mapping_counter), group_content
            )
            if re.search(r":\d+", group_mapped) is not None:
                mapping_counter += 1
            out.append("{" + group_mapped + "}")
            i = j + 1
        else:
            # try regex match
            m = atom_pattern.match(cgr, i)
            if m:
                token = m.group()
                # check case of uppercase+lowercase (like "Sc")
                if (
                    len(token) == 2
                    and token[0].isupper()
                    and token[1].islower()
                    and token not in ORGANIC_SUBSET
                ):
                    # split into separate atoms
                    # first uppercase
                    out.append(insert_mapping(token[0], mapping_counter))
                    mapping_counter += 1
                    # then lowercase as separate atom token
                    out.append(insert_mapping(token[1], mapping_counter))
                    mapping_counter += 1
                else:
                    out.append(insert_mapping(token, mapping_counter))
                    mapping_counter += 1
                i = m.end()
            else:
                out.append(cgr[i])
                i += 1
    return "".join(out)


def is_kekule(atom_mapped_rxn_smi: str) -> bool:
    """Check if a iven RXN SMILES is kekulized.

    Return True if all bracketed atoms in a mapped reaction SMILES string
    use Kekulé (uppercase) element symbols, i.e. no lowercase aromatic
    symbols like [c], [n], [o], etc. are present. Returns False otherwise.
    """
    for match in re.finditer(r"\[([^\]]+)\]", atom_mapped_rxn_smi):
        content = match.group(1)
        m = re.search(r"[A-Za-z]", content)
        if m:
            first = m.group(0)
            if not first.isupper():
                return False
    return True


def _rebuild_side_from_cgr(
    side_smi: str,
    scaffold: str,
    cgr_side_name: str,
    kekulized: bool,
) -> str:
    """Rebuild either reactant or product molecule from CGR scaffold."""
    try:
        parsed_bonds = parse_bonds_from_smiles(side_smi)

        # which bonds to delete:
        map_nums_unspecified_bonds = [key for key, val in parsed_bonds.items() if val == "~"]

        # Build RDKit Mol
        mol = Chem.MolFromSmiles(side_smi.replace("~", ""), sanitize=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ADJUSTHS)

        # Delete unspecified bonds
        mol = remove_bonds_by_atom_map_nums(mol, map_nums_unspecified_bonds)

        # Fix cis/trans stereo
        mol = update_cis_trans_stereo_chem(mol, parsed_bonds)

        # Convert back to SMILES
        smi = Chem.MolToSmiles(mol, canonical=False, kekuleSmiles=kekulized)

        # Fix chirality
        chiral_map_nums = get_chiral_center_map_nums(mol)
        smi = update_chirality_tags(smi, scaffold, chiral_map_nums)

        return smi

    except Exception as e:
        logger.warning(f"Failed to process CGR-SMILES {cgr_side_name} side '{side_smi}'. Error: {e}")
        return ""


def cgr_to_rxn(cgr_smiles: str, add_atom_mapping: bool = False) -> str:
    """Converts a CGR SMILES string back into a reaction SMILES string.

    This function reverses a Condensed Graph of Reaction (CGR) SMILES representation
    into standard reaction SMILES (`reactants>>products`). It reconstructs reactant
    and product molecules by removing unspecified bonds, updating stereochemistry,
    and restoring chirality tags based on the CGR annotations.

    Args:
        cgr_smiles (str): A CGR SMILES string representing a reaction, where changes
            between reactants and products are encoded using `{reac|prod}` syntax.
        add_atom_mapping (bool, optional): If True, ensures atom mappings are
            present in the output RXN SMILES. If False, atom mappings are stripped
            unless they were already present in the input. Default is False.

    Returns:
        str: The corresponding reaction SMILES string in the format "reactants>>products".

    Notes:
        - Each substitution pattern in the CGR SMILES should follow `{...|...}`.
        - Unspecified bonds (labeled as "~") are removed in the resulting molecules.
        - Stereochemistry and chirality tags are preserved and corrected during reconstruction.
        - This function is the reverse transformation of `rxn_to_cgr`.
    """
    if cgr_smiles == "":
        return ""

    try:
        if not is_cgr_smiles_fully_atom_mapped(cgr_smiles):
            cgr_smi = add_atom_mapping_to_cgr(cgr_smiles)
            input_atom_mapped = False
        else:
            cgr_smi = cgr_smiles
            input_atom_mapped = True

        kekulized = is_kekule(cgr_smi)

        # extract reac and prod smiles scaffold from cgr smiles
        reac_smi, prod_smi = get_reac_prod_scaffold_smiles_from_cgr(cgr_smi)
        cgr_reac_scaffold = reac_smi.replace("~", "")
        cgr_prod_scaffold = prod_smi.replace("~", "")

        # try each side independently
        reac_smi_final = _rebuild_side_from_cgr(reac_smi, cgr_reac_scaffold, "reactant", kekulized)
        prod_smi_final = _rebuild_side_from_cgr(prod_smi, cgr_prod_scaffold, "product", kekulized)

        # If everything failed entirely, fall back
        if reac_smi_final == "" and prod_smi_final == "":
            return ""

        rxn_smiles = f"{reac_smi_final}>>{prod_smi_final}"

        if not input_atom_mapped and not add_atom_mapping:
            rxn_smiles = remove_atom_mapping(rxn_smiles)
            rxn_smiles = remove_redundant_brackets(rxn_smiles)

        return rxn_smiles

    except Exception as e:
        logger.warning(
            f"Total failure in cgr_to_rxn for input '{cgr_smiles}'. " f"Error: {e}. Returning empty string."
        )
        return ""

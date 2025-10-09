import re
from collections import Counter
from typing import List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem

from cgr_smiles.logger import logger
from cgr_smiles.reaction_balancing import balance_reaction, is_rxn_balanced
from cgr_smiles.utils import (
    ORGANIC_SUBSET,
    TokenType,
    _tokenize,
    flip_e_z_stereo,
    get_atom_by_map_num,
    get_atom_map_adjacency_list_from_smiles,
    is_num_permutations_even,
    make_mol,
    map_reac_to_prod,
    remove_atom_mapping,
    update_all_atom_stereo,
)


def remove_redundant_brackets_and_hydrogens(cgr: str) -> str:
    """Remove redundant square brackets and explicit hydrogens from a CGR-SMILES string.

    This function cleans a CGR-SMILES string by removing brackets that contain only atoms
    from the ORGANIC_SUBSET and by eliminating explicit hydrogen atoms where possible,
    while preserving charges, isotopes, and other annotations.

    Args:
        cgr (str): A CGR-SMILES string potentially containing redundant brackets and explicit hydrogens.

    Returns:
        str: The cleaned CGR-SMILES string with redundant brackets and hydrogens removed.
    """
    # Special explicit-H patterns first
    specials = {
        "CH4": "C",
        "CH3": "C",
        "CH2": "C",
        "CH": "C",
        "OH2": "O",
        "OH": "O",
        # "oH": "o",
        "NH3": "N",
        "NH2": "N",
        "NH": "N",
        # "nH": "n",
        "SH2": "S",
        "SH": "S",
        # "sH": "s",
        "PH3": "P",
        "PH2": "P",
        "PH": "P",
        "cH": "c",
    }

    def replace_bracketed(match):
        atom = match.group(1)
        if atom in specials:
            return specials[atom]
        if atom in ORGANIC_SUBSET:
            return atom
        return f"[{atom}]"

    # replace all [..] with cleaned version
    cgr = re.sub(r"\[([^\]]+)\]", replace_bracketed, cgr)

    # collapse {X|X} → X
    cgr = re.sub(r"\{([A-Za-z0-9@+\-]+)\|\1\}", r"\1", cgr)

    return cgr


def remove_redundant_brackets(cgr: str) -> str:
    """Remove redundant square brackets from a CGR-SMILES string.

    Only brackets containing atoms from the ORGANIC_SUBSET are removed.
    Brackets that include explicit hydrogens, charges, isotopes, or other annotations are preserved.

    Args:
        cgr (str): A CGR-SMILES string potentially containing redundant brackets.

    Returns:
        str: The CGR-SMILES string with redundant brackets removed.
    """

    def replace_bracketed(match):
        atom = match.group(1)
        if atom in ORGANIC_SUBSET:
            return atom
        return f"[{atom}]"

    # Replace [X] with X if X is in organic subset
    cgr = re.sub(r"\[([^\]]+)\]", replace_bracketed, cgr)

    return cgr


def remove_aromatic_bonds(cgr_smiles: str) -> str:
    """Remove ':' used as bond descriptors, while preserving ':' used for atom mapping inside []."""
    result = []
    in_brackets = False
    for char in cgr_smiles:
        if char == "[":
            in_brackets = True
        elif char == "]":
            in_brackets = False

        # drop ':' if outside brackets (bond), else keep it
        if char == ":" and not in_brackets:
            continue
        result.append(char)

    return "".join(result)


class RxnToCgr:
    """Transform reaction SMILES into CGR SMILES.

    This class provides a callable interface to convert reaction SMILES into
    Condensed Graph of Reaction (CGR) SMILES. It supports single strings, lists
    of strings, pandas Series, and pandas DataFrames.

    The transformation can optionally adjust atom mappings, simplify SMILES
    notation, remove explicit hydrogens, balance reactions, and control whether
    aromatic systems are represented in Kekulé form.

    Attributes:
        keep_atom_mapping (bool): Whether to preserve atom mapping numbers
            in the output CGR SMILES.
        remove_brackets (bool): Whether to remove redundant square brackets
            from the SMILES.
        remove_hydrogens (bool): Whether to remove explicit hydrogen atoms
            from the SMILES.
        balance_rxn (bool): Whether to attempt balancing of the reaction
            stoichiometry before generating the CGR.
        rxn_col (Optional[str]): Column name in a DataFrame containing reaction SMILES.
        kekulize (bool): If True, converts aromatic atoms/bonds into an explicit
            Kekulé representation. If False (default), aromatic notation is kept.
        keep_aromatic_bonds (bool): Controls how aromaticity is handled when
            `kekulize=True`. If True, aromatic bonds are explicitly flagged in
            the Kekulé-expanded CGR. If False, aromaticity is fully expanded into
            alternating single/double bonds. Has no effect if `kekulize=False`.
    """

    def __init__(
        self,
        keep_atom_mapping: bool = False,
        remove_brackets: bool = False,
        remove_hydrogens: bool = False,
        balance_rxn: bool = False,
        rxn_col: Optional[str] = None,
        kekulize: bool = False,
        keep_aromatic_bonds: bool = True,
    ) -> None:
        """Initialize the transformation object.

        Args:
            keep_atom_mapping (bool, optional): If True, preserve atom mapping
                in the output. Defaults to False.
            remove_brackets (bool, optional): If True, remove square brackets
                from the SMILES where possible. Defaults to False.
            remove_hydrogens (bool, optional): If True, remove explicit hydrogens
                from the SMILES. Defaults to False.
            balance_rxn (bool, optional): If True, attempts to balance the reaction
                stoichiometry before generating the CGR. Defaults to False.
            rxn_col (str, optional): Column name in a DataFrame containing
                reaction SMILES. Required if passing a DataFrame. Defaults to None.
            kekulize (bool, optional): If True, converts all aromatic atoms/bonds into a
                Kekulé form with explicit single/double bonds. Defaults to False
                (keep aromatic SMILES notation).
            keep_aromatic_bonds (bool, optional): Effective only when `kekulize=True`.
                If True, aromatic bonds are explicitly retained in the Kekulé-expanded
                CGR (where supported). If False, aromaticity is fully expressed as
                alternating single/double bonds. Has no effect if `kekulize=False`.
                Defaults to True.
        """
        self.keep_atom_mapping = keep_atom_mapping
        self.remove_brackets = remove_brackets
        self.remove_hydrogens = remove_hydrogens
        self.balance_rxn = balance_rxn
        self.rxn_col = rxn_col
        self.kekulize = kekulize
        self.keep_aromatic_bonds = keep_aromatic_bonds

    def __call__(
        self, data: Union[str, List[str], pd.Series, pd.DataFrame]
    ) -> Union[str, List[str], pd.Series, pd.DataFrame]:
        """Apply the transformation to reaction SMILES.

        Args:
            data (Union[str, List[str], pd.Series, pd.DataFrame]): Input data
                containing reaction SMILES. Can be a single string, a list of
                strings, a pandas Series, or a pandas DataFrame.

        Returns:
            Union[str, List[str], pd.Series, pd.DataFrame]: Transformed CGR SMILES
            in the same structure as the input.

        Raises:
            ValueError: If a DataFrame is provided but `self.rxn_col` is not set.
            TypeError: If the input type is not supported.
        """
        if isinstance(data, str):
            return rxn_to_cgr(
                data,
                keep_atom_mapping=self.keep_atom_mapping,
                remove_brackets=self.remove_brackets,
                remove_hydrogens=self.remove_hydrogens,
                balance_rxn=self.balance_rxn,
                kekulize=self.kekulize,
                keep_aromatic_bonds=self.keep_aromatic_bonds,
            )

        elif isinstance(data, list):
            result = [self(d) for d in data]
            n_empty = sum(1 for item in result if item == "")
            if len(result) > 0:
                logger.warning(
                    f"Failed for {n_empty} out of {len(result)} samples ({n_empty / len(result) * 100} %). "
                )
            return result

        elif isinstance(data, pd.Series):
            return data.apply(self)

        elif isinstance(data, pd.DataFrame):
            if self.rxn_col is None:
                raise ValueError(
                    f"A pandas DataFrame was provided, but `self.rxn_col` is not set.\n"
                    f"Available columns are: {list(data.columns)}\n"
                    "Please specify the column name containing the reactions by setting "
                    "`rxn_col` at time of initialization."
                )
            return data[self.rxn_col].apply(self)

        else:
            raise TypeError("Input must be str, list, pandas Series, or DataFrame.")


def mask_nonshared_with_neg1_once_indices(list1, list2):
    """Replace elements not shared (one-to-one) between two lists with -1.

    and also return the indices where valid matches were kept. TODO.

    Returns:
    -------
    new1, new2 : list
        Updated lists with unmatched elements set to -1.
    idx1, idx2 : list of int
        Indices of matched elements in list1 and list2 respectively.
    """
    c1, c2 = Counter(list1), Counter(list2)
    shared_counts = {k: min(c1[k], c2[k]) for k in c1 if k in c2}

    # --- Process first list ---
    remaining = shared_counts.copy()
    new1, idx1 = [], []
    for i, x in enumerate(list1):
        if remaining.get(x, 0) > 0:
            new1.append(x)
            idx1.append(i)
            remaining[x] -= 1
        else:
            new1.append(-1)

    # --- Process second list ---
    remaining = shared_counts.copy()
    new2, idx2 = [], []
    for i, x in enumerate(list2):
        if remaining.get(x, 0) > 0:
            new2.append(x)
            idx2.append(i)
            remaining[x] -= 1
        else:
            new2.append(-1)

    return new1, new2, idx1, idx2


def move_matches_to_indices(list2, idx_from, idx_to):
    """Move elements in list2 from positions idx_from to target positions idx_to.

    Non-matched elements are shifted right (wrapping around if needed).

    Parameters
    ----------
    list2 : list
        Source list to rearrange (copy made automatically).
    idx_from : list of int
        Positions in list2 containing the matched elements.
    idx_to : list of int
        Target positions in the new list.

    Returns:
    -------
    new_list2 : list
        Rearranged copy of list2.
    """
    n = len(list2)
    new_list = list2.copy()
    # remove in reverse order so later indices stay valid
    items = [new_list[i] for i in idx_from]
    for i in sorted(idx_from, reverse=True):
        del new_list[i]

    # pad to full length with placeholders
    for val, tgt in sorted(zip(items, idx_to), key=lambda x: x[1]):
        new_list.insert(tgt, val)
        # ensure we don’t exceed original length
        if len(new_list) > n:
            new_list.pop()

    return new_list


def chunk_by_neighbors(atom_maps, center, nbrs):
    """TODO."""
    chunks = []
    current = []
    for m in atom_maps:
        if m in nbrs:
            if current != []:
                chunks.append(current)
            current = [m]
        elif m == center:
            if current != []:
                chunks.append(current)
            chunks.append([m])
            current = []
        else:
            current.append(m)

    if current:
        chunks.append(current)
    return chunks


def get_mol_cgr(mol_reac: Chem.Mol, mol_prod: Chem.Mol) -> Chem.Mol:
    """Build a molecule based on the reactant and prodcut molecules.

    Build the superimposed CGR molecule by combining the bonds of a reactant
    and product RDKit molecule using atom map correspondence.

    For each pair of atoms that are bonded in the product but not in the
    reactant, an unspecified bond is added to the reactant structure.
    The returned molecule thus contains all original reactant bonds plus
    additional placeholder bonds representing new or changed connections
    that appear in the product.

    Args:
        mol_reac (Mol): RDKit molecule of the reactant (atom mapped).
        mol_prod (Mol): RDKit molecule of the product (atom mapped)
            whose atom map numbers correspond to those in ``mol_reac``.

    Returns:
        Mol: A new molecule representing the CGR. This is the reactant
        molecule with additional unspecified bonds for any connections
        that appear in the product but are missing in the reactant.
    """
    ri2pi = map_reac_to_prod(mol_reac, mol_prod)

    # add missing bonds to the cgr mol
    mol_cgr = Chem.EditableMol(mol_reac)
    n_atoms = mol_reac.GetNumAtoms()
    unspecified_bonds = []
    for idx1 in range(n_atoms):
        for idx2 in range(idx1 + 1, n_atoms):
            bond_reac = mol_reac.GetBondBetweenAtoms(idx1, idx2)
            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[idx1], ri2pi[idx2])
            if bond_reac is None and bond_prod is not None:
                mol_cgr.AddBond(idx1, idx2, order=Chem.rdchem.BondType.UNSPECIFIED)
                unspecified_bonds.append((idx1, idx2))
    mol_cgr = mol_cgr.GetMol()
    return mol_cgr


def get_chirality_aligned_smiles_and_mols(
    rxn_smi: str, kekulize: bool
) -> Tuple[str, str, str, str, Chem.Mol, Chem.Mol, Chem.Mol]:
    """Build reactant, product, and CGR molecules from a reaction SMILES.

    Parses a reaction SMILES, builds RDKit molecule objects, and reorders
    them so that atom mapping and tetrahedral stereochemistry (chirality)
    are consistent between reactant and product sides.

    The routine performs the following steps:
      1. Parses the reaction SMILES into separate reactant and product strings.
      2. Builds RDKit Mol objects with optional kekulization.
      3. Constructs a condensed graph of the reaction (CGR) and uses its
         atom mapping to reorder reactant and product atoms.
      4. Detects tetrahedral centers present in both molecules and checks
         neighbor permutation parity.
      5. If an odd permutation is detected, reorders fragments in the reactant
         SMILES to restore consistent stereochemistry.
      6. Returns the possibly updated reaction SMILES along with intermediate
         SMILES strings and Mol objects.

    Args:
        rxn_smi (str): Reaction SMILES string with atom mappings.
        kekulize (bool): Whether to kekulize molecules when creating them.

    Returns:
        Tuple[str, str, str, str, Chem.Mol, Chem.Mol, Chem.Mol]:
            A tuple of:
                * **rxn_smi_aligned** (`str`): Possibly reordered reaction SMILES.
                * **smi_reac** (`str`): Reactant SMILES after any reordering.
                * **smi_prod** (`str`): Product SMILES after any reordering.
                * **smi_cgr** (`str`): SMILES of the condensed graph of reaction (CGR).
                * **mol_reac** (`Chem.Mol`): Reactant molecule.
                * **mol_prod** (`Chem.Mol`): Product molecule.
                * **mol_cgr** (`Chem.Mol`): CGR molecule.
    """
    for i in range(2):
        smi_reac, _, smi_prod = rxn_smi.split(">")
        mol_reac, mol_prod = (
            make_mol(smi_reac, kekulize=kekulize),
            make_mol(smi_prod, kekulize=kekulize),
        )

        # add missing bonds to the cgr mol
        mol_cgr = get_mol_cgr(mol_reac, mol_prod)
        smi_cgr = Chem.MolToSmiles(mol_cgr, canonical=False)
        mol_cgr = make_mol(smi_cgr, sanitize=False, kekulize=kekulize)

        # reorder reac and prod molecule so we get the relative stereochemistry tags right:
        # TODO: by doing the reordering, we basically canonicalize and make it a non-injective mapping
        # TODO: maybe instead just align the mapping of the product with the one in the reactant
        prod_map_to_id = dict([(atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol_prod.GetAtoms()])
        prod_reorder = [prod_map_to_id[a.GetAtomMapNum()] for a in mol_cgr.GetAtoms()]
        mol_prod = Chem.RenumberAtoms(mol_prod, prod_reorder)
        smi_prod = Chem.MolToSmiles(mol_prod, canonical=False)

        reac_map_to_id = dict([(atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol_reac.GetAtoms()])
        reac_reorder = [reac_map_to_id[a.GetAtomMapNum()] for a in mol_cgr.GetAtoms()]
        mol_reac = Chem.RenumberAtoms(mol_reac, reac_reorder)
        smi_reac = Chem.MolToSmiles(mol_reac, canonical=False)

        for atom in mol_reac.GetAtoms():
            chiral_tag = atom.GetChiralTag()

            if chiral_tag in (
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            ):
                map_num = atom.GetAtomMapNum()
                atom_prod = get_atom_by_map_num(mol_prod, map_num)
                chiral_tag_prod = atom_prod.GetChiralTag()

                if chiral_tag_prod in (
                    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                ):
                    adj_reac = get_atom_map_adjacency_list_from_smiles(smi_reac)
                    adj_prod = get_atom_map_adjacency_list_from_smiles(smi_prod)

                    # check if a neighbor changed.
                    nbrs_reac = adj_reac[map_num]
                    nbrs_prod = adj_prod[map_num]
                    a, b, idx1, idx2 = mask_nonshared_with_neg1_once_indices(nbrs_reac, nbrs_prod)
                    if -1 not in a and -1 not in b:
                        continue

                    flips_even = is_num_permutations_even(a, b)
                    if not flips_even:
                        frag_reac = smi_reac.split(".")
                        smi_reac = ".".join([frag_reac[-1]] + frag_reac[:-1])

                        rxn_smi = f"{smi_reac}>>{smi_prod}"

                        break

    return rxn_smi, smi_reac, smi_prod, smi_cgr, mol_reac, mol_prod, mol_cgr


# TODO: do some checks according to our assumptions (atom mapping, balanced etc)
# TODO: also make version that has the {..|..} for all atoms and bonds (not only those changing)
# TODO: standardize atom order depending on unmapped reactants, then add mappings again. Make the cgr molecule from this canonicalized reactant molecule to get maximum reproducibility  # noqa: E501
# TODO: Make this also work for unbalanced rxns
# DONE: als make unmapped version of the cgrsmiles
def rxn_to_cgr(
    rxn_smi: str,
    keep_atom_mapping: bool = False,
    remove_brackets: bool = False,
    remove_hydrogens: bool = False,
    balance_rxn: bool = False,
    kekulize: bool = False,
    keep_aromatic_bonds: bool = True,
) -> str:
    """Converts a reaction SMILES string into a Condensed Graph of Reaction (CGR) SMILES.

    A CGR SMILES encodes the transformation between reactant and product molecules
    as a single, compact string representation, where atoms and bonds are annotated to
    show differences in atom types, bond orders, and stereochemistry.

    Args:
        rxn_smi (str): A reaction SMILES string in the format "reactant>>product".
        keep_atom_mapping (bool): If True, atom map numbers will be removed in the
            output CGR SMILES. Otherwise they will be retained (default).
        remove_brackets (bool): If True, redundant square brackets will be removed
            in the output CGR SMILES. Otherwise they will be kept (default).
        remove_hydrogens (bool): If True, explicit hydrogens will be removed in the
            output CGR SMILES. Otherwise they will be kept (default).
        balance_rxn (bool, optional): If True, attempts to balance the reaction
            before generating the CGR. Defaults to False.
        kekulize (bool, optional): If True, converts all aromatic atoms/bonds into a
            specific Kekulé representation with alternating single/double bonds.
            Defaults to False (keep aromatic notation).
        keep_aromatic_bonds (bool, optional): If True and used together with
            `kekulize=False`, aromatic bonds will be explicitly retained in the
            Kekulé-expanded CGR (where supported). If False under `kekulize=False`,
            aromaticity is fully converted into alternating single/double bonds.
            Has no effect if `kekulize=True`. Defaults to True.

    Returns:
        str: A CGR SMILES string representing the reaction as a single molecule
        with annotations of changes using `{reac|prod}` syntax.

    Notes:
        - Requires all atoms in the SMILES to be atom-mapped.
        - Requires balanced reactions.

    Example:
        >>> smi_reac = "[C:1]([H:3])([H:4])([H:5])[H:6].[Cl:2][H:7]"
        >>> smi_prod = "[C:1]([H:3])([H:4])([H:5])[Cl:2].[H:6][H:7]"
        >>> rxn_smiles = f"{smi_reac}>>{smi_prod}"
        >>> rxn_to_cgr(rxn_smiles)
        "[C:1]1([H:3])([H:4])([H:5]){-|~}[H:6]{~|-}[H:7]{-|~}[Cl:2]{~|-}1"

        # In the resulting CGR SMILES, the `{reac|prod}` notation encodes how atoms and bonds
        # change from reactants to products. For example, '[H:6]{~|-}[H:7]' means that while there
        # was no bond between these two hydrogen atoms in the reactants, a single bond has been
        # formed between them in the product molecule.
    """
    try:
        if not is_rxn_balanced(rxn_smi):
            if balance_rxn:
                rxn_smi = balance_reaction(rxn_smi, kekulize=kekulize)
            else:
                raise ValueError(
                    "The given rxn is not balanced. To enable cgr transform, set `balance_reaction=True`."
                )

        # TODO: check if rxn_smiles is atom mapped, if not, add mapping.
        # TODO: maybe let this function make the assumption of balanced, fully mapped reactions.
        # Handling of the preparation, shall the wrapper do.
        # fully_atom_mapped = is_fully_atom_mapped(rxn_smi)
        # if not fully_atom_mapped:
        #     print(f"WARNING: given reaction smiles is not fully atom mapped: {rxn_smi}")
        #     rxn_smi = add_atom_mapping(rxn_smi)

        rxn_smi, smi_reac, smi_prod, smi_cgr, mol_reac, mol_prod, mol_cgr = (
            get_chirality_aligned_smiles_and_mols(rxn_smi, kekulize)
        )
        n_atoms = mol_reac.GetNumAtoms()

        update_all_atom_stereo(mol_reac, smi_reac, smi_cgr)
        update_all_atom_stereo(mol_prod, smi_prod, smi_cgr)

        replace_dict_atoms = {}
        replace_dict_bonds = {}

        for i1 in range(n_atoms):
            atom_reac = mol_reac.GetAtomWithIdx(i1)
            atom_cgr = mol_cgr.GetAtomWithIdx(i1)
            atom_prod = mol_prod.GetAtomWithIdx(i1)

            reac_smarts = atom_reac.GetSmarts(isomericSmiles=True)
            prod_smarts = atom_prod.GetSmarts(isomericSmiles=True)

            if reac_smarts != prod_smarts:
                replace_dict_atoms[atom_cgr.GetAtomMapNum()] = f"{{{reac_smarts}|{prod_smarts}}}"
            else:
                replace_dict_atoms[atom_cgr.GetAtomMapNum()] = reac_smarts

            for i2 in range(i1 + 1, n_atoms):
                atom2_cgr = mol_cgr.GetAtomWithIdx(i2)
                map_num_1 = atom_cgr.GetAtomMapNum()
                map_num_2 = atom2_cgr.GetAtomMapNum()
                bond_reac = mol_reac.GetBondBetweenAtoms(i1, i2)
                bond_prod = mol_prod.GetBondBetweenAtoms(i1, i2)

                reac_begin, reac_end = map_num_1, map_num_2
                smarts_bond_reac = "~"
                if bond_reac is not None:
                    smarts_bond_reac = bond_reac.GetSmarts(allBondsExplicit=True)

                    reac_begin, reac_end = (
                        bond_reac.GetBeginAtom().GetAtomMapNum(),
                        bond_reac.GetEndAtom().GetAtomMapNum(),
                    )

                prod_begin, prod_end = map_num_1, map_num_2
                smarts_bond_prod = "~"
                if bond_prod is not None:
                    smarts_bond_prod = bond_prod.GetSmarts(allBondsExplicit=True)
                    prod_begin, prod_end = (
                        bond_prod.GetBeginAtom().GetAtomMapNum(),
                        bond_prod.GetEndAtom().GetAtomMapNum(),
                    )

                    if (
                        reac_begin == prod_end and reac_end == prod_begin
                    ):  # TODO: maybe actually compare to cgr begin end atom, not reac vs. prod.
                        # need to flip!
                        smarts_bond_prod = flip_e_z_stereo(smarts_bond_prod)

                if bond_reac is None and bond_prod is None:
                    continue

                if smarts_bond_reac != smarts_bond_prod:
                    val = f"{{{smarts_bond_reac}|{smarts_bond_prod}}}"
                else:
                    val = smarts_bond_reac if smarts_bond_reac != "-" else ""

                replace_dict_bonds[(reac_begin, reac_end)] = val
                replace_dict_bonds[(reac_end, reac_begin)] = flip_e_z_stereo(val)

        # change bonds
        smiles = ""
        anchor = None
        idx = 0
        next_bond = None
        branches = []
        ring_nums = {}
        i2m = {}
        for tokentype, token_idx, token in _tokenize(smi_cgr):
            if tokentype == TokenType.ATOM:
                i2m[idx] = int(token[:-1].split(":")[1])
                if anchor is not None:
                    if next_bond is None:
                        next_bond = ""
                    smiles += replace_dict_bonds.get((i2m[anchor], i2m[idx]), next_bond)
                    next_bond = None
                smiles += token
                anchor = idx
                idx += 1
            elif tokentype == TokenType.BRANCH_START:
                branches.append(anchor)
                smiles += token
            elif tokentype == TokenType.BRANCH_END:
                anchor = branches.pop()
                smiles += token
            elif tokentype == TokenType.BOND_TYPE:
                next_bond = token
            elif tokentype == TokenType.EZSTEREO:
                next_bond = token
            elif tokentype == TokenType.RING_NUM:
                if token in ring_nums:
                    jdx, order = ring_nums[token]
                    if next_bond is None and order is None:
                        next_bond = ""
                    elif order is None:
                        next_bond = next_bond
                    elif next_bond is None:
                        next_bond = order
                    smiles += replace_dict_bonds.get((i2m[idx - 1], i2m[jdx]), next_bond)
                    smiles += str(token)
                    next_bond = None
                    del ring_nums[token]

                else:
                    ring_nums[token] = (idx - 1, next_bond)
                    next_bond = None
                    smiles += str(token)

        smi_cgr = smiles

        # change atoms
        for k in replace_dict_atoms.keys():
            smi_cgr = smi_cgr.replace(re.findall(rf"\[[^):]*:{k}\]", smi_cgr)[0], replace_dict_atoms[k])

        if not keep_atom_mapping:
            smi_cgr = remove_atom_mapping(smi_cgr)

        if remove_brackets and remove_hydrogens:
            smi_cgr = remove_redundant_brackets_and_hydrogens(smi_cgr)

        elif remove_brackets:
            smi_cgr = remove_redundant_brackets(smi_cgr)

        if not kekulize and not keep_aromatic_bonds:
            smi_cgr = remove_aromatic_bonds(smi_cgr)

        return smi_cgr

    except Exception as e:
        logger.warning(f"Failed to process RXN-SMILES '{rxn_smi}'. Error: {e}. Returning empty string.")
        return ""

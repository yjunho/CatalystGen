import os
import glob
import numpy as np
import itertools
import json
from pymatgen.core import Structure
from smact import element_dictionary, metals
from smact.screening import pauling_test
from tqdm import tqdm

# 평가 대상
adsorbates = ['CHCO', 'CHOCHO', 'NO2NO2', 'COCH2O']
root_dir = "/home/kaist/projects/CatalystGen-main/generated2"
min_distance_threshold = 0.8


def smact_validity(elem_symbols, count, use_pauling_test=True, include_alloys=True):
    try:
        space = element_dictionary(elem_symbols)
        smact_elems = [e[1] for e in space.items()]
        electronegs = [e.pauling_eneg for e in smact_elems]
        ox_combos = [e.oxidation_states for e in smact_elems]
    except:
        return False

    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys and all(e in metals for e in elem_symbols):
        return True

    threshold = max(count)

    for ox_states in itertools.product(*ox_combos):
        total_charge = sum([ox * c for ox, c in zip(ox_states, count)])
        if abs(total_charge) < 1e-3:
            if use_pauling_test:
                try:
                    if pauling_test(ox_states, electronegs):
                        return True
                except TypeError:
                    return True
            else:
                return True
    return False


results = {}

for ads in adsorbates:
    print(f"\nEvaluating {ads}...")
    cif_dir = os.path.join(root_dir, ads, "cif")
    cif_files = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))

    total = len(cif_files)
    valid_struct = 0
    valid_comp = 0

    parse_fail = 0
    volume_fail = 0
    distance_fail = 0
    smact_fail = 0

    for path in tqdm(cif_files):
        try:
            structure = Structure.from_file(path)
        except Exception:
            parse_fail += 1
            continue

        if structure.volume <= 0:
            volume_fail += 1
            continue
        dmat = structure.distance_matrix
        min_dist = dmat[np.triu_indices_from(dmat, k=1)].min()
        if min_dist < min_distance_threshold:
            distance_fail += 1
            continue

        valid_struct += 1

        atom_counts = structure.composition.get_el_amt_dict()
        elem_symbols = list(atom_counts.keys())
        count = list(atom_counts.values())

        if smact_validity(elem_symbols, count):
            valid_comp += 1
        else:
            smact_fail += 1

    results[ads] = {
        "total": total,
        "valid_structural": valid_struct,
        "valid_compositional": valid_comp,
        "parse_fail": parse_fail,
        "volume_fail": volume_fail,
        "distance_fail": distance_fail,
        "smact_fail": smact_fail,
    }

# 출력
print("\n======= Final Validity Summary (CDVAE Style) =======")
for ads in adsorbates:
    print(f"\n[ {ads} ]")
    for k, v in results[ads].items():
        print(f"{k:>25}: {v}")

# JSON 저장
save_path = os.path.join(root_dir, "validity_results.json")
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n📁 Validity results saved to: {save_path}")

"""
fix_taxon_groups.py — correct misfiled organisms in the BENDER group labels (2026-09-27).

A genus-by-genus review of BENDER_BIO.csv found organisms filed in the wrong group:
  * 286 parasitic flatworms and nematodes, 2 plant-parasitic nematodes, a fish and a sea urchin
    filed as Protists, Plants or Fungi. They are animals but not mammals, so they move to their
    curation group (Nematodes, Other_invertebrates, Other_vertebrates) and leave the six groups.
  * Aspergillus oryzae and Pyricularia oryzae (fungi) filed as Plants        -> Fungi
  * Mycolicibacterium paratuberculosis, Candidatus Phytoplasma solani (bacteria)
    filed as Fungi                                                            -> Bacteria
  * Stemona tuberosa (a plant) filed as Fungi                                 -> Plants

Updates BENDER_BIO.csv `kingdom` and BENDER_300K_lineages.csv `lineage` (animals -> Animals).
Trajectory folders are not moved. Dry run by default; --apply writes, after copying each file to
<name>_before_taxonfix.csv beside it (never overwritten).
"""

import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared modules (bender_common, ...)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # this folder (also flat on HPC)
import bender_common as bc  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

BIO = str(paths.BIO)
LINEAGES = str(paths.LINEAGES)

FLATWORMS = ['Schistosoma', 'Protopolystoma', 'Schistocephalus', 'Macrostomum', 'Echinostoma', 'Fasciola',
             'Mesocestoides', 'Dibothriocephalus', 'Hydatigena', 'Rodentolepis']
NEMATODES = ['Meloidogyne', 'Caenorhabditis', 'Romanomermis', 'Bursaphelenchus', 'Panagrolaimus', 'Teladorsagia',
             'Ancylostoma', 'Brugia', 'Nippostrongylus', 'Globodera', 'Acrobeloides', 'Trichuris', 'Gongylonema',
             'Heligmosomoides', 'Plectus', 'Strongylus', 'Diploscapter', 'Onchocerca', 'Trichinella', 'Haemonchus',
             'Loa', 'Mesorhabditis', 'Toxocara', 'Ditylenchus', 'Cylicostephanus', 'Litomosoides', 'Steinernema',
             'Parascaris', 'Soboliphyme', 'Dracunculus', 'Pristionchus', 'Panagrellus', 'Strongyloides',
             'Heterodera', 'Enterobius', 'Heterorhabditis', 'Aphelenchoides']
# genus -> (curation group for BENDER_BIO `kingdom`, lineage for BENDER_300K_lineages.csv)
RULES = {g: ('Other_invertebrates', 'Animals') for g in FLATWORMS}
RULES.update({g: ('Nematodes', 'Animals') for g in NEMATODES})
RULES.update({'Scomberomorus': ('Other_vertebrates', 'Animals'),
              'Heliocidaris': ('Other_invertebrates', 'Animals'),
              'Aspergillus oryzae': ('Fungi', 'Fungi'),
              'Pyricularia oryzae': ('Fungi', 'Fungi'),
              'Mycolicibacterium paratuberculosis': ('Bacteria', 'Bacteria'),
              'Candidatus Phytoplasma': ('Bacteria', 'Bacteria'),
              'Stemona tuberosa': ('Plants', 'Plants')})


def target(organism):
    """(kingdom, lineage) for an organism covered by a rule, else None. Longest matching prefix wins."""
    if not isinstance(organism, str):
        return None
    hits = [k for k in RULES if organism == k or organism.startswith(k + ' ')]
    return RULES[max(hits, key=len)] if hits else None


def main(apply):
    bio = pd.read_csv(BIO, low_memory=False)
    lin = pd.read_csv(LINEAGES)
    t_bio = bio['organism'].map(target)
    t_lin = lin['organism'].map(target)
    new_k = [t[0] if t else k for t, k in zip(t_bio, bio['kingdom'])]
    new_l = [t[1] if t else l for t, l in zip(t_lin, lin['lineage'])]
    changed = pd.DataFrame({'old': bio['kingdom'], 'new': new_k})
    changed = changed[changed['old'] != changed['new']]
    print(f'BENDER_BIO: {len(changed)} rows change kingdom')
    print(changed.value_counts().to_string())
    lc = pd.DataFrame({'old': lin['lineage'], 'new': new_l})
    lc = lc[lc['old'] != lc['new']]
    print(f'\nBENDER_300K_lineages: {len(lc)} rows change lineage')
    print(lc.value_counts().to_string())
    six = pd.Series(new_k).isin(bc.SIX_LINEAGES)
    print(f'\nsix groups after the fix: {int(six.sum())} sequences')
    print(pd.Series(new_k)[six].value_counts().to_string())
    if not apply:
        print('dry run: nothing written (pass --apply)')
        return
    for path in (BIO, LINEAGES):
        b = path.replace('.csv', '_before_taxonfix.csv')
        if not os.path.exists(b):
            shutil.copy2(path, b)
    bio['kingdom'] = new_k
    lin['lineage'] = new_l
    bio.to_csv(BIO, index=False)
    lin.to_csv(LINEAGES, index=False)
    for path in (BIO, LINEAGES):
        bc.write_provenance(path, __file__, {'step': 'misfiled organisms reassigned (genus review)',
                                             'rules': {k: list(v) for k, v in RULES.items()}},
                            [path.replace('.csv', '_before_taxonfix.csv')])
    print('written; backups *_before_taxonfix.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    main(ap.parse_args().apply)

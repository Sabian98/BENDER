"""
ncbi_taxonomy_check.py — verify BENDER group labels against NCBI Taxonomy (2026-09-27).

Fetches the full lineage of every taxonomy ID in BENDER_300K_lineages.csv from the NCBI E-utilities
(efetch, db=taxonomy; batches of 200, < 3 requests/s, only taxonomy IDs are sent), caches it in
ncbi_taxonomy_cache.json, and classifies each organism:
  Viruses / Bacteria / Archaea      by domain (superkingdom)
  Mammals                           Mammalia
  Nematodes, Insects, Other_vertebrates, Other_invertebrates   other Metazoa
  Fungi                             Fungi (NCBI includes Microsporidia)
  Plants                            Viridiplantae (land plants and green algae)
  Algae                             other Archaeplastida (red algae, glaucophytes)
  Protists                          every other eukaryote
Compares with BENDER_BIO.csv `kingdom` (all rows) and the lineage table `lineage`, and writes the
disagreements to ncbi_taxonomy_mismatches.csv. Report only; nothing is changed.
"""

import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # analysis/
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

HERE = str(paths.TAXONOMY.parent)   # cache and outputs
CACHE = os.path.join(HERE, 'ncbi_taxonomy_cache.json')
OUT = os.path.join(HERE, 'ncbi_taxonomy_mismatches.csv')
BIO = str(paths.BIO)
LINEAGES = str(paths.LINEAGES)
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'


def fetch(taxids):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [t for t in taxids if str(t) not in cache]
    for k in range(0, len(todo), 200):
        batch = todo[k:k + 200]
        data = urllib.parse.urlencode({'db': 'taxonomy', 'id': ','.join(map(str, batch)), 'retmode': 'xml',
                                       'tool': 'bender_taxcheck'}).encode()
        xml = urllib.request.urlopen(urllib.request.Request(EUTILS, data=data), timeout=120).read()
        for tx in ET.fromstring(xml).findall('Taxon'):
            rec = {'name': tx.findtext('ScientificName'), 'rank': tx.findtext('Rank'),
                   'lineage': [(t.findtext('Rank'), t.findtext('ScientificName')) for t in tx.findall('LineageEx/Taxon')]}
            ids = [tx.findtext('TaxId')] + [a.text for a in tx.findall('AkaTaxIds/TaxId')]
            for i in ids:
                cache[str(i)] = rec
        json.dump(cache, open(CACHE, 'w'))
        print(f'  fetched {min(k + 200, len(todo))}/{len(todo)}', flush=True)
        time.sleep(0.5)
    return cache


def classify(rec):
    if rec is None:
        return 'not_found'
    names = {n for _, n in rec['lineage']} | {rec['name']}
    if 'Viruses' in names:
        return 'Viruses'
    if 'Bacteria' in names:
        return 'Bacteria'
    if 'Archaea' in names:
        return 'Archaea'
    if 'Eukaryota' not in names:
        return 'Unclassified'
    if 'Mammalia' in names:
        return 'Mammals'
    if 'Metazoa' in names:
        if 'Nematoda' in names:
            return 'Nematodes'
        if 'Insecta' in names:
            return 'Insects'
        if 'Vertebrata' in names:
            return 'Other_vertebrates'
        return 'Other_invertebrates'
    if 'Fungi' in names:
        return 'Fungi'
    if 'Viridiplantae' in names:
        return 'Plants'
    if 'Rhodophyta' in names or 'Glaucocystophyceae' in names or 'Rhodophyceae' in names:
        return 'Algae'
    return 'Protists'


def main():
    lin = pd.read_csv(LINEAGES)
    bio = pd.read_csv(BIO, usecols=['UniProt_ID', 'kingdom', 'organism', 'taxonomy_id'], low_memory=False)
    taxids = sorted(set(lin['taxid'].dropna().astype(int)) | set(bio['taxonomy_id'].dropna().astype(int)))
    print(f'{len(taxids)} taxonomy IDs')
    cache = fetch(taxids)
    lin['ncbi_group'] = lin['taxid'].map(lambda t: classify(cache.get(str(int(t)))) if pd.notna(t) else 'no_taxid')
    lin['ncbi_name'] = lin['taxid'].map(lambda t: (cache.get(str(int(t))) or {}).get('name') if pd.notna(t) else None)
    ANIMAL = {'Mammals', 'Nematodes', 'Insects', 'Other_vertebrates', 'Other_invertebrates'}
    lin['ncbi_lineage'] = lin['ncbi_group'].map(lambda g: 'Animals' if g in ANIMAL else g)
    m = bio.merge(lin[['uniprot_id', 'ncbi_group', 'ncbi_name']], left_on='UniProt_ID', right_on='uniprot_id', how='left')
    six = ['Mammals', 'Fungi', 'Plants', 'Protists', 'Bacteria', 'Viruses']
    # a six-group label must equal the NCBI group; an out-of-dataset label must at least not be a six group
    bad_bio = m[(m['kingdom'].isin(six) | m['ncbi_group'].isin(six)) & (m['kingdom'] != m['ncbi_group'])]
    print(f'\nBENDER_BIO: {len(bio)} rows; disagreements with NCBI: {len(bad_bio)}')
    print(bad_bio.groupby(['kingdom', 'ncbi_group']).size().to_string() if len(bad_bio) else '  none')
    bad_lin = lin[lin['lineage'] != lin['ncbi_lineage']]
    print(f'\nlineage table: {len(lin)} rows; disagreements with NCBI: {len(bad_lin)}')
    print(bad_lin.groupby(['lineage', 'ncbi_lineage']).size().to_string() if len(bad_lin) else '  none')
    cols = ['UniProt_ID', 'organism', 'taxonomy_id', 'kingdom', 'ncbi_group', 'ncbi_name']
    bad_bio[cols].to_csv(OUT, index=False)
    lin.to_csv(os.path.join(HERE, 'ncbi_taxonomy_lineages.csv'), index=False)
    print(f'\nwrote {OUT} and ncbi_taxonomy_lineages.csv')
    print(bad_bio[cols].groupby(['kingdom', 'ncbi_group'])['organism'].agg(lambda x: '; '.join(sorted(set(x))[:6])).to_string()
          if len(bad_bio) else '')


if __name__ == '__main__':
    main()

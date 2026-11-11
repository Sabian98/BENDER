"""Find BENDER rice (Oryza sativa subsp. japonica) entries that are no longer rice gene models.

UniProt deleted many Oryza sativa entries in January 2026 as "not part of a reference proteome". That reason alone is
administrative, so an entry is excluded only when it is both no longer active in UniProtKB and shares no 20-residue
segment with any protein of the current rice reference proteome (UniProt UP000059680).

Inputs
  bio.csv                      from prepare_inputs.py (all 10,282 BENDER sequences)
  UP000059680.fasta            the rice reference proteome, downloaded with
                               curl -L -o UP000059680.fasta.gz \
                                 "https://rest.uniprot.org/uniprotkb/stream?format=fasta&compressed=true&query=(proteome:UP000059680)"
Outputs (in <outdir>)
  rice_ref_match.tsv           one row per BENDER rice entry: active in UniProtKB, matches the reference proteome
  exclude_rice.txt             accessions to exclude (inactive and no match)

Usage: python rice_ref_match.py <bio.csv> <UP000059680.fasta> <outdir>
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

K = 20


def read_fasta(path):
    seqs, cur = [], []
    for line in open(path):
        if line.startswith('>'):
            if cur:
                seqs.append(''.join(cur))
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append(''.join(cur))
    return seqs


def active_accessions(accs, batch=100):
    """Return the subset of accessions that UniProtKB still serves as active entries."""
    active = set()
    for i in range(0, len(accs), batch):
        q = ' OR '.join(f'accession:{a}' for a in accs[i:i + batch])
        url = 'https://rest.uniprot.org/uniprotkb/search?' + urllib.parse.urlencode({'query': q, 'fields': 'accession', 'format': 'json', 'size': 500})
        for attempt in range(3):
            try:
                res = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'bender-rice-check/1.0'}), timeout=60))
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError('UniProt query failed')
        active.update(r['primaryAccession'] for r in res.get('results', []) if not str(r.get('entryType', '')).startswith('Inactive'))
    return active


def main(bio, fasta, out):
    d = pd.read_csv(bio, low_memory=False)
    # only subspecies japonica, the subspecies the reference proteome covers; indica entries cannot be judged against it
    rice = d[d.organism.fillna('') == 'Oryza sativa subsp. japonica'][['UniProt_ID', 'organism', 'sequence', 'nu_ok_cal', 'nu_ok_rech', 'nu_orig', 'nu']].copy()
    ref = read_fasta(fasta)
    kmers = set()
    for s in ref:
        kmers.update(s[i:i + K] for i in range(len(s) - K + 1))
    rice['matches_reference'] = [any(s[i:i + K] in kmers for i in range(len(s) - K + 1)) for s in rice.sequence]
    act = active_accessions(rice.UniProt_ID.tolist())
    rice['active_in_uniprot'] = rice.UniProt_ID.isin(act)
    rice['exclude'] = ~rice.active_in_uniprot & ~rice.matches_reference
    rice['in_paper_set'] = rice.nu_ok_cal.astype(bool) & rice.nu_ok_rech.astype(bool) & rice.nu_orig.notna() & rice.nu.notna()
    os.makedirs(out, exist_ok=True)
    rice.drop(columns=['sequence', 'nu_ok_cal', 'nu_ok_rech', 'nu_orig', 'nu']).to_csv(os.path.join(out, 'rice_ref_match.tsv'), sep='\t', index=False)
    rice.loc[rice.exclude, 'UniProt_ID'].sort_values().to_csv(os.path.join(out, 'exclude_rice.txt'), index=False, header=False)
    for lab, s in [('all BENDER', rice), ('paper set', rice[rice.in_paper_set])]:
        ina = s[~s.active_in_uniprot]
        print(f'{lab}: {len(s)} rice entries, {len(ina)} no longer active, of which {(~ina.matches_reference).sum()} '
              f'({100 * (~ina.matches_reference).mean():.0f}%) share no {K}-residue segment with the reference proteome; '
              f'active entries matching: {100 * s[s.active_in_uniprot].matches_reference.mean():.0f}%; excluded {s.exclude.sum()}')
    print(f'reference proteome: {len(ref)} proteins')


if __name__ == '__main__':
    main(*sys.argv[1:4])

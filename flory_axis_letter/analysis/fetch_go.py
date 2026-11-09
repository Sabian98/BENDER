import sys, csv, io, time, urllib.request, pandas as pd
src, out = sys.argv[1], sys.argv[2]
ids = pd.read_csv(src, usecols=['UniProt_ID']).UniProt_ID.dropna().unique().tolist()
rows=[]
for i in range(0, len(ids), 100):
    url=("https://rest.uniprot.org/uniprotkb/accessions?accessions="+",".join(ids[i:i+100])+
         "&fields=accession,protein_name,go_p,go_f,go_c,keyword,protein_families&format=tsv&size=100")
    for a in range(4):
        try:
            txt=urllib.request.urlopen(url, timeout=60).read().decode(); break
        except Exception as e:
            time.sleep(3)
    rows += list(csv.DictReader(io.StringIO(txt), delimiter='\t'))
pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
print(len(ids), 'queried', len(rows), 'returned')

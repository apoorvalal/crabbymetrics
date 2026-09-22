"""Checksum-locked 49-study archive; extract inputs/source, not generated graphics."""
from pathlib import Path
import hashlib
import json
import re
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent; CACHE=ROOT/'.cache'
URL='https://dataverse.harvard.edu/api/access/datafile/11026794'
MD5='2cc3608ba63eee17701fc5b47270dbc1'
p=CACHE/'reanalysis.zip'
if not p.exists():
    with urllib.request.urlopen(URL,timeout=240) as src, p.open('wb') as dst:
        while chunk:=src.read(1024*1024): dst.write(chunk)
assert hashlib.file_digest(p.open('rb'),'md5').hexdigest()==MD5
z=zipfile.ZipFile(p)
selected=[i for i in z.infolist() if not i.is_dir() and (
    i.filename.startswith(('replication/code/','replication/rawdata/','replication/metadata/'))
    or (i.filename.startswith('replication/markdown/') and i.filename.endswith('.Rmd'))
    or i.filename=='replication/output/summary.RData')]
for i in selected:
    dst=(CACHE/i.filename).resolve(); assert dst.is_relative_to(CACHE.resolve())
    if not dst.exists(): dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(z.read(i))
jobs=[]
for source in sorted((CACHE/'replication/markdown').glob('*/*.Rmd')):
    blocks=re.findall(r'```\{r[^\n]*\}\n(.*?)```',source.read_text(errors='replace'),re.S)
    first=next(b for b in blocks if 'readRDS(' in b)
    setup=re.split(r'^(?:[\w.]+\s*(?:<-|=)\s*)?panelview\(',first,flags=re.M)[0]
    # These setup snippets were reviewed at the pinned archive revision. They
    # read the shipped data and set columns/IDs; no plotting or estimator code.
    point=next((b for b in blocks if re.search(r'twfe\.point\.est\.2\s*<-\s*twfe\.point\.coef',b)), '')
    extra=re.search(r'FE_plus\s*=\s*(c\([^)]*\)|["\'][^"\']+["\'])',point)
    extras=re.findall(r'["\']([^"\']+)["\']',extra.group(1)) if extra else []
    trends=bool(re.search(r'ULT\s*=\s*(1|TRUE)',point))
    jobs.append(dict(name=source.parent.name,source=str(source.relative_to(CACHE)),setup=setup,
                     extra_effects=extras,unit_trends=trends,point_call=point.strip()))
(CACHE/'archive-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
(ROOT/'results/archive-provenance.json').write_text(json.dumps(dict(
    doi='10.7910/DVN/9RJFZF',version='1.0',url=URL,bytes=p.stat().st_size,md5=MD5,
    sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),studies=len(jobs)),indent=2)+'\n')
print('Verified archive; prepared',len(jobs),'study specifications')

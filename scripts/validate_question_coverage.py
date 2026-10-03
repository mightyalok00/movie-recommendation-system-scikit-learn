"""Validate exact MovieLens question coverage Q1-Q105."""
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [(1,10),(11,22),(23,31),(32,39),(40,49),(50,57),(58,63),(64,71),(72,83),(84,95),(96,105)]

def main():
    ranges=[]
    for path in sorted((ROOT/'solutions').glob('[0-9][0-9]_*.py')):
        match=re.search(r'Answers Questions (\d+) through (\d+)', path.read_text(encoding='utf-8'))
        if not match: raise SystemExit(f'Missing question range: {path}')
        ranges.append((int(match.group(1)),int(match.group(2))))
    if ranges != EXPECTED: raise SystemExit(f'Expected {EXPECTED}, found {ranges}')
    covered=[q for s,e in ranges for q in range(s,e+1)]
    if covered != list(range(1,106)): raise SystemExit('Coverage is not contiguous Q1-Q105')
    print('Question coverage verified: exactly Q1-Q105 across 11 sections.')

if __name__=='__main__': main()

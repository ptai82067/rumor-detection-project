"""Runtime validation script for Streamlit UI."""
import urllib.request, sys, time

BASE = 'http://localhost:8503'
pages = [
    ('Home', '/'),
    ('Research Evolution', '/Research_Evolution'),
    ('Rumor Detection', '/Rumor_Detection'),
    ('Ontology KG Explorer', '/Ontology_KG_Explorer'),
    ('Feature Analysis', '/Feature_Analysis'),
    ('Experimental Results', '/Experimental_Results'),
]

results = []

# Test 1: Server reachable
for _ in range(5):
    try:
        r = urllib.request.urlopen(f'{BASE}/', timeout=5)
        results.append(('Server reachable', 'PASS', f'HTTP {r.status}'))
        break
    except:
        time.sleep(1)
else:
    results.append(('Server reachable', 'FAIL', 'Not responding on 8503'))

# Test 2: All pages
for name, path in pages:
    try:
        r = urllib.request.urlopen(f'{BASE}{path}', timeout=15)
        size = len(r.read())
        s = 'PASS' if r.status == 200 else 'FAIL'
        results.append((f'Page: {name}', s, f'HTTP {r.status} ({size:,} bytes)'))
    except Exception as e:
        results.append((f'Page: {name}', 'FAIL', str(e)[:80]))

# Print results
print('=' * 70)
print('RUNTIME VALIDATION RESULTS')
print('=' * 70)
pass_count = 0
for test, status, detail in results:
    icon = '[OK]' if status == 'PASS' else '[FAIL]'
    if status == 'PASS': pass_count += 1
    print(f'  {icon} {status}: {test} -> {detail}')

print(f'\n{"=" * 70}')
print(f'Results: {pass_count}/{len(results)} passed')
print(f'Server: {BASE}')
print(f'{"=" * 70}')
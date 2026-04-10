#!/usr/bin/env python
"""Debug script to investigate the 0 nodes/0 edges issue."""

from rdflib import Graph, Namespace

g = Graph()
g.parse('data/processed/pheme_kg.ttl', format='turtle')
EX = Namespace('http://example.org/pheme#')

# Check the first few repliesTo triples
print("Testing triple extraction...")
count = 0
for s, p, o in list(g.triples((None, EX.repliesTo, None)))[:1]:
    uri_str_s = str(s)
    uri_str_o = str(o)
    print(f'uri_str_s: {repr(uri_str_s)}')
    print(f'uri_str_o: {repr(uri_str_o)}')
    
    has_post_s = '/post/' in uri_str_s
    has_post_o = '/post/' in uri_str_o
    print(f"'/post/' in uri_str_s: {has_post_s}")
    print(f"'/post/' in uri_str_o: {has_post_o}")
    count += 1

print(f"Checked {count} triples")

# Now test the full extraction
import networkx as nx

G = nx.DiGraph()
edge_count = 0
for subject, predicate, obj in g.triples((None, EX.repliesTo, None)):
    uri_str_s = str(subject)
    uri_str_o = str(obj)
    
    # Extract post IDs
    if '/post/' in uri_str_s and '/post/' in uri_str_o:
        try:
            child_id = int(uri_str_s.split('/post/')[-1].split('#')[0].split('?')[0])
            parent_id = int(uri_str_o.split('/post/')[-1].split('#')[0].split('?')[0])
            G.add_edge(parent_id, child_id)
            edge_count += 1
        except ValueError:
            pass

print(f'\nEdges added: {edge_count}')
print(f'Nodes in graph: {G.number_of_nodes()}')
print(f'Edges in graph: {G.number_of_edges()}')
if G.number_of_nodes() > 0:
    print(f'First few nodes: {list(G.nodes())[:5]}')
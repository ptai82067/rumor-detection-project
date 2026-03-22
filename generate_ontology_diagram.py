#!/usr/bin/env python3
"""
Generate ontology diagram from PHEME ontology TTL file
"""

import rdflib
from rdflib import Namespace, RDF, RDFS, OWL
import graphviz
import os

def load_ontology(file_path):
    """Load the ontology from TTL file"""
    g = rdflib.Graph()
    g.parse(file_path, format='turtle')
    return g

def extract_ontology_structure(g):
    """Extract classes, properties, and relationships from the ontology"""
    
    # Define namespaces
    EX = Namespace("http://example.org/pheme#")
    
    # Extract classes
    classes = []
    for s in g.subjects(RDF.type, OWL.Class):
        label = g.value(s, RDFS.label) or str(s).split('#')[-1]
        comment = g.value(s, RDFS.comment) or ""
        classes.append({
            'uri': s,
            'label': str(label),
            'comment': str(comment)
        })
    
    # Extract object properties
    object_properties = []
    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        label = g.value(s, RDFS.label) or str(s).split('#')[-1]
        domain = g.value(s, RDFS.domain)
        range_ = g.value(s, RDFS.range)
        comment = g.value(s, RDFS.comment) or ""
        object_properties.append({
            'uri': s,
            'label': str(label),
            'domain': domain,
            'range': range_,
            'comment': str(comment)
        })
    
    # Extract data properties
    data_properties = []
    for s in g.subjects(RDF.type, OWL.DatatypeProperty):
        label = g.value(s, RDFS.label) or str(s).split('#')[-1]
        domain = g.value(s, RDFS.domain)
        range_ = g.value(s, RDFS.range)
        comment = g.value(s, RDFS.comment) or ""
        data_properties.append({
            'uri': s,
            'label': str(label),
            'domain': domain,
            'range': str(range_).split('#')[-1] if range_ else "xsd:datatype",
            'comment': str(comment)
        })
    
    return classes, object_properties, data_properties

def create_class_diagram(classes, object_properties, data_properties):
    """Create a class diagram using Graphviz"""
    
    dot = graphviz.Digraph(comment='PHEME Ontology Diagram', format='png')
    dot.attr(rankdir='TB', size='12,8', dpi='300')
    dot.attr('node', shape='record', style='rounded,filled', fillcolor='lightblue')
    
    # Create class nodes
    class_nodes = {}
    for cls in classes:
        class_id = str(cls['uri']).split('#')[-1]
        class_nodes[class_id] = cls['uri']
        
        # Create record shape with attributes
        label_parts = [f"{{<head>{cls['label']}|"]
        
        # Add object properties for this class
        obj_attrs = []
        for prop in object_properties:
            if prop['domain'] == cls['uri']:
                range_label = str(prop['range']).split('#')[-1] if prop['range'] else 'Unknown'
                obj_attrs.append(f"{prop['label']}: {range_label}")
        
        # Add data properties for this class
        data_attrs = []
        for prop in data_properties:
            if prop['domain'] == cls['uri']:
                data_attrs.append(f"{prop['label']}: {prop['range']}")
        
        # Combine attributes
        all_attrs = obj_attrs + data_attrs
        if all_attrs:
            label_parts.extend(all_attrs)
        else:
            label_parts.append("No properties")
        
        label_parts.append("}}")
        label = "|".join(label_parts)
        
        dot.node(class_id, label=label, tooltip=cls['comment'])
    
    # Create property edges
    for prop in object_properties:
        if prop['domain'] and prop['range']:
            domain_id = str(prop['domain']).split('#')[-1]
            range_id = str(prop['range']).split('#')[-1]
            
            if domain_id in class_nodes and range_id in class_nodes:
                dot.edge(domain_id, range_id, label=prop['label'], arrowhead='open')
    
    return dot

def main():
    """Main function to generate the ontology diagram"""
    
    # Path to ontology file
    ontology_file = "ontology/pheme_ontology_v1.ttl"
    
    if not os.path.exists(ontology_file):
        print(f"Error: Ontology file not found at {ontology_file}")
        return
    
    print("Loading ontology...")
    g = load_ontology(ontology_file)
    
    print("Extracting ontology structure...")
    classes, object_properties, data_properties = extract_ontology_structure(g)
    
    print(f"Found {len(classes)} classes")
    print(f"Found {len(object_properties)} object properties")
    print(f"Found {len(data_properties)} data properties")
    
    print("Generating diagram...")
    dot = create_class_diagram(classes, object_properties, data_properties)
    
    # Render the diagram
    output_file = "ontology_diagram"
    print(f"Rendering diagram to {output_file}...")
    dot.render(output_file, view=True, cleanup=True)
    
    print(f"Diagram generated successfully: {output_file}.png")

if __name__ == "__main__":
    main()
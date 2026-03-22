#!/usr/bin/env python3
"""
Generate Mermaid diagram from PHEME ontology TTL file
"""

import rdflib
from rdflib import Namespace, RDF, RDFS, OWL
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

def create_mermaid_diagram(classes, object_properties, data_properties):
    """Create a Mermaid class diagram"""
    
    mermaid_lines = [
        "```mermaid",
        "classDiagram",
        "    class Event {",
        "        +String label",
        "        +String comment",
        "    }",
        "    class Post {",
        "        +String text",
        "        +DateTime createdAt",
        "        +Integer depth",
        "        +Integer childrenCount",
        "        +Float timeSinceSource",
        "    }",
        "    class User {",
        "        +String label",
        "        +String comment",
        "    }",
        "    class ConversationThread {",
        "        +Integer threadSize",
        "        +Integer maxDepth",
        "        +Float replySpeed",
        "    }",
        "    class VeracityLabel {",
        "        +String label",
        "        +String comment",
        "    }",
        "    class NonRumor {",
        "        +String label",
        "        +String comment",
        "    }",
        "    class Rumor {",
        "        +String label",
        "        +String comment",
        "    }",
        "",
        "    Post --> User : postedBy",
        "    Post --> Event : aboutEvent",
        "    Post --> Post : repliesTo",
        "    Post --> ConversationThread : inThread",
        "    ConversationThread --> VeracityLabel : hasVeracity",
        "    VeracityLabel <|-- NonRumor",
        "    VeracityLabel <|-- Rumor",
        "```"
    ]
    
    return "\n".join(mermaid_lines)

def create_mermaid_er_diagram(classes, object_properties, data_properties):
    """Create a Mermaid ER diagram"""
    
    mermaid_lines = [
        "```mermaid",
        "erDiagram",
        "    Event {",
        "        string label",
        "        string comment",
        "    }",
        "    Post {",
        "        string text",
        "        datetime createdAt",
        "        integer depth",
        "        integer childrenCount",
        "        float timeSinceSource",
        "    }",
        "    User {",
        "        string label",
        "        string comment",
        "    }",
        "    ConversationThread {",
        "        integer threadSize",
        "        integer maxDepth",
        "        float replySpeed",
        "    }",
        "    VeracityLabel {",
        "        string label",
        "        string comment",
        "    }",
        "    NonRumor {",
        "        string label",
        "        string comment",
        "    }",
        "    Rumor {",
        "        string label",
        "        string comment",
        "    }",
        "",
        "    Post ||--|| User : postedBy",
        "    Post ||--|| Event : aboutEvent",
        "    Post ||--|| Post : repliesTo",
        "    Post ||--|| ConversationThread : inThread",
        "    ConversationThread ||--|| VeracityLabel : hasVeracity",
        "    VeracityLabel ||--|| NonRumor : is_a",
        "    VeracityLabel ||--|| Rumor : is_a",
        "```"
    ]
    
    return "\n".join(mermaid_lines)

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
    
    print("Generating Mermaid diagrams...")
    
    # Generate class diagram
    class_diagram = create_mermaid_diagram(classes, object_properties, data_properties)
    with open("ontology_mermaid_class_diagram.md", "w", encoding="utf-8") as f:
        f.write("# PHEME Ontology - Class Diagram\n\n")
        f.write(class_diagram)
    
    # Generate ER diagram
    er_diagram = create_mermaid_er_diagram(classes, object_properties, data_properties)
    with open("ontology_mermaid_er_diagram.md", "w", encoding="utf-8") as f:
        f.write("# PHEME Ontology - ER Diagram\n\n")
        f.write(er_diagram)
    
    print("Mermaid diagrams generated successfully:")
    print("- ontology_mermaid_class_diagram.md")
    print("- ontology_mermaid_er_diagram.md")
    print("\nYou can view these diagrams using any Mermaid-compatible viewer or tool.")

if __name__ == "__main__":
    main()
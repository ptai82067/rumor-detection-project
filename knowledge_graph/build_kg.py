"""
Knowledge Graph Construction for PHEME Rumor Detection

This module converts the PHEME dataset features into RDF triples
using the minimal ontology (Version 1) designed for rumor detection.

Input: data/processed/pheme_features.csv
Output: data/processed/pheme_kg.ttl
"""

import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD
import logging
from typing import Dict, Set
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define namespaces
EX = Namespace("http://example.org/pheme#")
RDF = RDF
RDFS = RDFS
OWL = OWL
XSD = XSD


class KnowledgeGraphBuilder:
    """Builds a knowledge graph from PHEME dataset using the minimal ontology."""
    
    def __init__(self):
        self.graph = Graph()
        self.graph.bind("ex", EX)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)
        
        # Track created entities to avoid duplicates
        self.created_posts: Set[int] = set()
        self.created_users: Set[int] = set()
        self.created_events: Set[str] = set()
        self.created_threads: Set[int] = set()
        self.created_labels: Set[int] = set()
        
        # Track valid post_ids for repliesTo validation
        self.valid_post_ids: Set[int] = set()
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load the processed PHEME features dataset."""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert time to datetime if not already done
        if df['time'].dtype == 'object':
            df['time'] = pd.to_datetime(df['time'])
        
        logger.info(f"Loaded {len(df)} posts across {df['thread_id'].nunique()} threads")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the dataset for consistency."""
        logger.info("Validating dataset consistency...")
        
        # Check for duplicate post_ids
        duplicate_posts = df[df.duplicated('post_id', keep=False)]
        if len(duplicate_posts) > 0:
            logger.error(f"Found {len(duplicate_posts)} duplicate post_ids!")
            return False
        
        # Check for missing required columns
        required_columns = ['post_id', 'user_id', 'text', 'time', 'event_id', 
                          'thread_id', 'label', 'depth', 'children_count']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for NaN values in critical columns
        critical_columns = ['post_id', 'user_id', 'event_id', 'thread_id', 'label']
        for col in critical_columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.error(f"Found {nan_count} NaN values in critical column: {col}")
                return False
        
        logger.info("✅ Dataset validation passed")
        return True
    
    def create_uri(self, entity_type: str, identifier) -> URIRef:
        """Create a URI for an entity based on its type and identifier."""
        if entity_type == 'post':
            return EX[f'post/{identifier}']
        elif entity_type == 'user':
            return EX[f'user/{identifier}']
        elif entity_type == 'event':
            return EX[f'event/{identifier}']
        elif entity_type == 'thread':
            return EX[f'thread/{identifier}']
        elif entity_type == 'label':
            return EX[f'label/{identifier}']
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")
    
    def add_class_instance(self, entity_uri: URIRef, class_uri: URIRef):
        """Add a class instance triple to the graph."""
        self.graph.add((entity_uri, RDF.type, class_uri))
    
    def add_object_property(self, subject_uri: URIRef, property_uri: URIRef, object_uri: URIRef):
        """Add an object property triple to the graph."""
        self.graph.add((subject_uri, property_uri, object_uri))
    
    def add_data_property(self, subject_uri: URIRef, property_uri: URIRef, value, datatype=None):
        """Add a data property triple to the graph."""
        if datatype:
            self.graph.add((subject_uri, property_uri, Literal(value, datatype=datatype)))
        else:
            self.graph.add((subject_uri, property_uri, Literal(value)))
    
    def process_post(self, row: pd.Series):
        """Process a single post and add its triples to the graph."""
        post_id = int(row['post_id'])
        
        # Skip if already processed
        if post_id in self.created_posts:
            return
        
        # Create URIs
        post_uri = self.create_uri('post', post_id)
        user_uri = self.create_uri('user', int(row['user_id']))
        event_uri = self.create_uri('event', row['event_id'])
        thread_uri = self.create_uri('thread', int(row['thread_id']))
        
        # Add class instance
        self.add_class_instance(post_uri, EX.Post)
        self.created_posts.add(post_id)
        
        # Add object properties
        self.add_object_property(post_uri, EX.postedBy, user_uri)
        self.add_object_property(post_uri, EX.aboutEvent, event_uri)
        self.add_object_property(post_uri, EX.inThread, thread_uri)
        
        # Add data properties
        self.add_data_property(post_uri, EX.text, str(row['text']))
        self.add_data_property(post_uri, EX.createdAt, row['time'].isoformat(), XSD.dateTime)
        self.add_data_property(post_uri, EX.depth, int(row['depth']), XSD.integer)
        self.add_data_property(post_uri, EX.childrenCount, int(row['children_count']), XSD.integer)
        self.add_data_property(post_uri, EX.timeSinceSource, float(row['time_since_source']), XSD.float)
        
        # Handle repliesTo relationship with validation
        reply_to = row['reply_to']
        depth = int(row['depth'])
        
        # Enforce consistency between depth and repliesTo
        if depth == 0:
            # Root post: repliesTo MUST be None
            if pd.notna(reply_to):
                logger.warning(f"Inconsistent data: post {post_id} has depth=0 but repliesTo={reply_to}. Removing incorrect repliesTo edge.")
                # Do NOT create the repliesTo relationship for root posts
            # No repliesTo edge should be created for depth=0
        else:
            # Non-root post: repliesTo MUST exist
            if pd.isna(reply_to):
                logger.error(f"Inconsistent data: post {post_id} has depth={depth} but no repliesTo. This should not happen in a proper tree structure.")
            elif int(reply_to) in self.valid_post_ids:
                # Target post exists, create the relationship
                reply_to_uri = self.create_uri('post', int(reply_to))
                self.add_object_property(post_uri, EX.repliesTo, reply_to_uri)
            else:
                # Target post does NOT exist, log warning and skip creating the triple
                logger.warning(f"Broken repliesTo relationship: post {post_id} references non-existent parent post {int(reply_to)}")
    
    def process_user(self, user_id: int):
        """Process a user and add its triples to the graph."""
        if user_id in self.created_users:
            return
        
        user_uri = self.create_uri('user', user_id)
        self.add_class_instance(user_uri, EX.User)
        self.created_users.add(user_id)
    
    def process_event(self, event_id: str):
        """Process an event and add its triples to the graph."""
        if event_id in self.created_events:
            return
        
        event_uri = self.create_uri('event', event_id)
        self.add_class_instance(event_uri, EX.Event)
        self.created_events.add(event_id)
    
    def process_thread(self, thread_id: int, thread_data: pd.DataFrame):
        """Process a conversation thread and add its triples to the graph."""
        if thread_id in self.created_threads:
            return
        
        thread_uri = self.create_uri('thread', thread_id)
        self.add_class_instance(thread_uri, EX.ConversationThread)
        self.created_threads.add(thread_id)
        
        # Add thread-level properties (using the first row as representative)
        first_row = thread_data.iloc[0]
        
        self.add_data_property(thread_uri, EX.threadSize, int(first_row['thread_size']), XSD.integer)
        self.add_data_property(thread_uri, EX.maxDepth, int(first_row['max_depth']), XSD.integer)
        self.add_data_property(thread_uri, EX.replySpeed, float(first_row['reply_speed_per_hour']), XSD.float)
        
        # Add veracity label
        label_value = int(first_row['label'])
        label_uri = self.create_uri('label', label_value)
        
        # Create label instance if not exists
        if label_value not in self.created_labels:
            if label_value == 0:
                self.add_class_instance(label_uri, EX.NonRumor)
            else:
                self.add_class_instance(label_uri, EX.Rumor)
            self.created_labels.add(label_value)
        
        self.add_object_property(thread_uri, EX.hasVeracity, label_uri)
    
    def build_knowledge_graph(self, df: pd.DataFrame):
        """Build the complete knowledge graph from the dataset."""
        logger.info("Building knowledge graph...")
        
        # Step 1: Collect all valid post_ids for repliesTo validation
        self.valid_post_ids = set(df['post_id'].unique())
        logger.info(f"Collected {len(self.valid_post_ids)} valid post_ids for repliesTo validation")
        
        # Step 2: Group by thread for efficient processing
        thread_groups = df.groupby('thread_id')
        
        for thread_id, thread_data in thread_groups:
            # Process thread first
            self.process_thread(thread_id, thread_data)
            
            # Process each post in the thread
            for _, post_row in thread_data.iterrows():
                self.process_post(post_row)
                
                # Process associated entities
                self.process_user(int(post_row['user_id']))
                self.process_event(post_row['event_id'])
        
        logger.info(f"Knowledge graph built with {len(self.graph)} triples")
        logger.info(f"Entities created: {len(self.created_posts)} posts, {len(self.created_users)} users, "
                   f"{len(self.created_events)} events, {len(self.created_threads)} threads")
    
    def detect_and_remove_cycles(self) -> bool:
        """Detect and remove cycles in the reply tree graph using DFS."""
        logger.info("Detecting and removing cycles in reply tree...")
        
        # Build adjacency list from repliesTo relationships
        reply_graph = {}
        reply_edges = []
        
        # Collect all repliesTo relationships
        for subject, predicate, obj in self.graph.triples((None, EX.repliesTo, None)):
            if predicate == EX.repliesTo:
                source_id = self.extract_post_id_from_uri(subject)
                target_id = self.extract_post_id_from_uri(obj)
                
                # Skip if we couldn't extract valid IDs
                if source_id is None or target_id is None:
                    continue
                
                if source_id not in reply_graph:
                    reply_graph[source_id] = []
                reply_graph[source_id].append(target_id)
                reply_edges.append((source_id, target_id))
        
        logger.info(f"Analyzing {len(reply_edges)} reply relationships for cycles")
        
        # Detect cycles using DFS
        visited = set()
        rec_stack = set()
        cycles_found = []
        
        def dfs_detect_cycle(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Check for self-loops
            if node_id in reply_graph and node_id in reply_graph[node_id]:
                cycles_found.append([node_id, node_id])
                logger.warning(f"Self-loop detected: post {node_id} → itself")
                # Remove self-loop
                reply_graph[node_id].remove(node_id)
                return True
            
            cycle_found = False
            if node_id in reply_graph:
                for neighbor_id in reply_graph[node_id][:]:  # Use copy to allow modification
                    if neighbor_id not in visited:
                        if dfs_detect_cycle(neighbor_id):
                            cycle_found = True
                    elif neighbor_id in rec_stack:
                        # Cycle detected
                        cycle_path = self.find_cycle_path(reply_graph, node_id, neighbor_id)
                        cycles_found.append(cycle_path)
                        logger.warning(f"Cycle detected: {' → '.join(map(str, cycle_path))} → {cycle_path[0]}")
                        
                        # Remove the problematic edge that closes the cycle
                        if neighbor_id in reply_graph[node_id]:
                            reply_graph[node_id].remove(neighbor_id)
                            logger.info(f"Removed cycle edge: {node_id} → {neighbor_id}")
                        cycle_found = True
            
            rec_stack.remove(node_id)
            return cycle_found
        
        # Run cycle detection on all nodes
        for node_id in reply_graph:
            if node_id not in visited:
                dfs_detect_cycle(node_id)
        
        if cycles_found:
            logger.warning(f"Found and removed {len(cycles_found)} cycles")
            # Rebuild the graph without cycles
            self.remove_cycle_edges_from_graph(reply_graph, reply_edges)
        else:
            logger.info("No cycles detected in reply tree")
        
        return len(cycles_found) == 0
    
    def extract_post_id_from_uri(self, uri: URIRef) -> int:
        """Extract post ID from URI string."""
        uri_str = str(uri)
        if '/post/' in uri_str:
            try:
                # Extract the part after '/post/' and before any fragment or query
                post_part = uri_str.split('/post/')[-1].split('#')[0].split('?')[0]
                return int(post_part)
            except ValueError:
                logger.warning(f"Could not extract post ID from URI: {uri_str}")
                return None
        elif 'http://example.org/pheme#post/' in uri_str:
            try:
                # Handle the case where the URI is in the format http://example.org/pheme#post/ID
                post_part = uri_str.split('http://example.org/pheme#post/')[-1].split('#')[0].split('?')[0]
                return int(post_part)
            except ValueError:
                logger.warning(f"Could not extract post ID from URI: {uri_str}")
                return None
        else:
            logger.warning(f"URI format not recognized: {uri_str}")
            return None
    
    def find_cycle_path(self, graph: dict, start_node: int, target_node: int) -> list:
        """Find the path that forms a cycle."""
        path = []
        visited = set()
        
        def dfs_path(node, target):
            if node == target:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            path.append(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    if dfs_path(neighbor, target):
                        return True
            
            path.pop()
            return False
        
        dfs_path(start_node, target_node)
        return path + [target_node]
    
    def remove_cycle_edges_from_graph(self, reply_graph: dict, original_edges: list):
        """Remove cycle edges from the RDF graph."""
        logger.info("Removing cycle edges from RDF graph...")
        
        # Remove all existing repliesTo relationships
        self.graph.remove((None, EX.repliesTo, None))
        
        # Re-add only the non-cyclic edges
        for source_id, target_id in original_edges:
            if source_id in reply_graph and target_id in reply_graph[source_id]:
                source_uri = self.create_uri('post', source_id)
                target_uri = self.create_uri('post', target_id)
                self.add_object_property(source_uri, EX.repliesTo, target_uri)
        
        logger.info("Cycle edges removed from RDF graph")
    
    def validate_graph(self) -> bool:
        """Validate the constructed knowledge graph."""
        logger.info("Validating knowledge graph...")
        
        # Check for duplicate post instances
        post_instances = list(self.graph.subjects(RDF.type, EX.Post))
        if len(post_instances) != len(self.created_posts):
            logger.error("Inconsistent post instance count!")
            return False
        
        # Check that each thread has exactly one veracity label
        thread_labels = {}
        for thread_uri in self.graph.subjects(RDF.type, EX.ConversationThread):
            labels = list(self.graph.objects(thread_uri, EX.hasVeracity))
            if len(labels) != 1:
                logger.error(f"Thread {thread_uri} has {len(labels)} veracity labels (should be exactly 1)")
                return False
            thread_labels[thread_uri] = labels[0]
        
        # Check repliesTo consistency (no cycles, proper tree structure)
        reply_relations = list(self.graph.triples((None, EX.repliesTo, None)))
        logger.info(f"Found {len(reply_relations)} reply relationships")
        
        # Detect and remove cycles
        cycles_removed = self.detect_and_remove_cycles()
        
        if not cycles_removed:
            logger.error("Failed to remove all cycles from the graph")
            return False
        
        # Final cycle verification
        if not self.verify_acyclic_graph():
            logger.error("Graph still contains cycles after removal")
            return False
        
        logger.info("✅ Knowledge graph validation passed")
        return True
    
    def verify_acyclic_graph(self) -> bool:
        """Final verification that the graph is acyclic."""
        logger.info("Performing final acyclic verification...")
        
        # Build graph for verification
        reply_graph = {}
        for subject, predicate, obj in self.graph.triples((None, EX.repliesTo, None)):
            if predicate == EX.repliesTo:
                source_id = self.extract_post_id_from_uri(subject)
                target_id = self.extract_post_id_from_uri(obj)
                
                if source_id not in reply_graph:
                    reply_graph[source_id] = []
                reply_graph[source_id].append(target_id)
        
        # Simple DFS to verify no cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            if node in reply_graph:
                for neighbor in reply_graph[node]:
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
            
            rec_stack.remove(node)
            return False
        
        for node in reply_graph:
            if node not in visited:
                if has_cycle(node):
                    logger.error(f"Cycle still exists in graph starting from node {node}")
                    return False
        
        logger.info("✅ Final verification: Graph is acyclic")
        return True
    
    def save_graph(self, output_file: str):
        """Save the knowledge graph to a Turtle file."""
        logger.info(f"Saving knowledge graph to {output_file}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Serialize to Turtle format
        self.graph.serialize(destination=output_file, format='turtle', encoding='utf-8')
        logger.info(f"Knowledge graph saved successfully to {output_file}")
    
    def build_complete_kg(self, input_file: str, output_file: str):
        """Complete pipeline: load data, build KG, validate, and save."""
        logger.info("Starting complete knowledge graph construction pipeline...")
        
        # Step 1: Load data
        df = self.load_data(input_file)
        
        # Step 2: Validate data
        if not self.validate_data(df):
            raise ValueError("Data validation failed")
        
        # Step 3: Build knowledge graph
        self.build_knowledge_graph(df)
        
        # Step 4: Validate graph
        if not self.validate_graph():
            raise ValueError("Graph validation failed")
        
        # Step 5: Save graph
        self.save_graph(output_file)
        
        logger.info("✅ Knowledge graph construction completed successfully!")


def main():
    """Main function to run the knowledge graph construction."""
    builder = KnowledgeGraphBuilder()
    
    input_file = "data/processed/pheme_features.csv"
    output_file = "data/processed/pheme_kg.ttl"
    
    try:
        builder.build_complete_kg(input_file, output_file)
        
        # Print summary statistics
        print(f"\n📊 Knowledge Graph Statistics:")
        print(f"   Total triples: {len(builder.graph)}")
        print(f"   Posts: {len(builder.created_posts)}")
        print(f"   Users: {len(builder.created_users)}")
        print(f"   Events: {len(builder.created_events)}")
        print(f"   Threads: {len(builder.created_threads)}")
        print(f"   Reply relationships: {len(list(builder.graph.triples((None, EX.repliesTo, None))))}")
        
    except Exception as e:
        logger.error(f"Error during knowledge graph construction: {str(e)}")
        raise


if __name__ == "__main__":
    main()
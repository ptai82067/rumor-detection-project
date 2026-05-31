"""
Knowledge Graph Construction for PHEME Rumor Detection — Version 2

This module extends Version 1 with:
  - Ontology v2 integration (pheme_ontology_v2.ttl)
  - Subclass typing (SourcePost, ReplyPost)
  - Extended post properties (postId, isSourcePost, textLength, etc.)
  - Graph centrality triples (pagerankScore, betweennessScore, etc.)
  - User properties (userPostCount, userPriorRumorRatio, etc.)
  - Event properties (eventName, eventType, etc.)
  - Thread properties (threadId, rumorLabel, belongsToEvent, hasSourcePost)
  - User-Thread participation links (participatesInThread)

Input: data/processed/pheme_features_with_graph.csv
       (fallback: data/processed/pheme_features.csv)
Output: data/processed/pheme_kg_v2.ttl
"""

import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD
import logging
from typing import Dict, Set, List, Tuple
import os
import math

# Set up logging to both console and file
log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'kg_build_v2.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Define namespaces
EX = Namespace("http://example.org/pheme#")
RDF = RDF
RDFS = RDFS
OWL = OWL
XSD = XSD

# Hardcoded event information for PHEME events
EVENT_INFO = {
    "charliehebdo":      {"name": "Charlie Hebdo Attack",       "type": "terrorism"},
    "sydneysiege":       {"name": "Sydney Cafe Siege",          "type": "terrorism"},
    "ferguson":          {"name": "Ferguson Unrest",            "type": "social_unrest"},
    "ottawashooting":    {"name": "Ottawa Parliament Shooting", "type": "terrorism"},
    "germanwings-crash": {"name": "Germanwings Flight 9525 Crash", "type": "accident"},
}


class KnowledgeGraphBuilderV2:
    """Builds a knowledge graph from PHEME dataset using the extended ontology v2."""
    
    def __init__(self):
        self.graph = Graph()
        self.graph.bind("ex", EX)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)
        
        # Load the ontology v2
        ontology_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ontology', 'pheme_ontology_v2.ttl'
        )
        if os.path.exists(ontology_path):
            self.graph.parse(ontology_path, format="turtle")
            logger.info(f"Loaded ontology from {ontology_path}")
        else:
            logger.warning(f"Ontology file not found at {ontology_path}, continuing without parsing")
        
        # Track created entities to avoid duplicates
        self.created_posts: Set[int] = set()
        self.created_users: Set[int] = set()
        self.created_events: Set[str] = set()
        self.created_threads: Set[int] = set()
        
        # Track valid post_ids for repliesTo validation
        self.valid_post_ids: Set[int] = set()
        
        # Track post metadata for structural validation: post_id -> (thread_id, depth)
        self.post_metadata: Dict[int, tuple] = {}
        
        # Track per-user aggregated data
        self.user_aggregates: Dict[int, dict] = {}
        
        # Track thread -> event mapping
        self.thread_to_event: Dict[int, str] = {}
        
        # Track thread -> source_post mapping
        self.thread_to_source_post: Dict[int, int] = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load the processed PHEME features dataset."""
        logger.info(f"Loading data from {file_path}")
        # Load reply_to as string to preserve precision for large tweet IDs
        # (float64 cannot precisely represent 18-digit tweet IDs)
        df = pd.read_csv(file_path, dtype={'reply_to': str})
        
        # Convert time to datetime if not already done
        if df['time'].dtype in ('object', 'string'):
            df['time'] = pd.to_datetime(df['time'])
        
        # Clean up reply_to: replace 'nan' strings with actual NaN
        df['reply_to'] = df['reply_to'].replace('nan', pd.NA)
        
        logger.info(f"Loaded {len(df)} posts across {df['thread_id'].nunique()} threads")
        logger.info(f"Columns available: {list(df.columns)}")
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
    
    def _precompute_user_aggregates(self, df: pd.DataFrame):
        """Precompute per-user aggregate statistics from the DataFrame."""
        logger.info("Precomputing user aggregate statistics...")
        
        # Group by user_id and compute aggregates
        user_groups = df.groupby('user_id')
        
        for user_id, group in user_groups:
            uid = int(user_id) if not isinstance(user_id, (int, float)) or math.isnan(user_id) else user_id
            if isinstance(uid, float) and math.isnan(uid):
                continue
            uid = int(uid)
            
            aggregates = {}
            
            # Compute from raw data
            aggregates['post_count'] = len(group)
            aggregates['thread_count'] = group['thread_id'].nunique()
            aggregates['avg_depth'] = float(group['depth'].mean())
            
            # User prior rumor ratio: ratio of posts in rumor threads
            rumor_posts = (group['label'] == 1).sum()
            aggregates['prior_rumor_ratio'] = float(rumor_posts / len(group)) if len(group) > 0 else 0.0
            
            # Source network size and credibility: computed per thread, use average
            # These will be computed per-thread and assigned to source users
            
            self.user_aggregates[uid] = aggregates
        
        logger.info(f"Precomputed aggregates for {len(self.user_aggregates)} users")
    
    def _precompute_thread_metadata(self, df: pd.DataFrame):
        """Precompute thread-level metadata (event mapping, source post)."""
        logger.info("Precomputing thread metadata...")
        
        for thread_id, group in df.groupby('thread_id'):
            tid = int(thread_id) if not isinstance(thread_id, (int, float)) or math.isnan(thread_id) else thread_id
            if isinstance(tid, float) and math.isnan(tid):
                continue
            tid = int(tid)
            
            # Get event_id from first row
            first_row = group.iloc[0]
            self.thread_to_event[tid] = str(first_row['event_id'])
            
            # Find source post (depth = 0 or is_source = True)
            source_candidates = group[group['depth'] == 0]
            if len(source_candidates) > 0:
                source_post = int(source_candidates.iloc[0]['post_id'])
                self.thread_to_source_post[tid] = source_post
            else:
                # Fallback: use first post chronologically
                if 'time' in group.columns:
                    earliest = group.loc[group['time'].idxmin()]
                    self.thread_to_source_post[tid] = int(earliest['post_id'])
                else:
                    self.thread_to_source_post[tid] = int(group.iloc[0]['post_id'])
        
        logger.info(f"Precomputed metadata for {len(self.thread_to_event)} threads")
    
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
    
    def _safe_add_data_property(self, subject_uri: URIRef, property_uri: URIRef, 
                                 value, datatype=None):
        """Safely add a data property, skipping if value is NaN or None."""
        if value is None:
            return
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return
        self.add_data_property(subject_uri, property_uri, value, datatype)
    
    def _safe_float(self, value) -> float:
        """Convert value to float, return NaN on failure."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return float('nan')
    
    def _safe_int(self, value) -> int:
        """Convert value to int, return 0 on failure."""
        try:
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return 0
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def process_post(self, row: pd.Series, df_columns: set):
        """Process a single post and add its triples to the graph (v2 extended)."""
        post_id = int(row['post_id'])
        
        # Skip if already processed
        if post_id in self.created_posts:
            return
        
        # Create URIs
        post_uri = self.create_uri('post', post_id)
        user_uri = self.create_uri('user', int(row['user_id']))
        event_uri = self.create_uri('event', row['event_id'])
        thread_uri = self.create_uri('thread', int(row['thread_id']))
        
        # Add class instance (base type)
        self.add_class_instance(post_uri, EX.Post)
        
        # Add subclass typing (v2)
        depth = int(row['depth'])
        is_source = False
        if 'is_source' in df_columns:
            is_source = bool(row['is_source'])
        
        if depth == 0 or is_source:
            self.add_class_instance(post_uri, EX.SourcePost)
        else:
            self.add_class_instance(post_uri, EX.ReplyPost)
        
        self.created_posts.add(post_id)
        
        # Add object properties (v1)
        self.add_object_property(post_uri, EX.postedBy, user_uri)
        self.add_object_property(post_uri, EX.aboutEvent, event_uri)
        self.add_object_property(post_uri, EX.inThread, thread_uri)
        
        # Add data properties (v1)
        self.add_data_property(post_uri, EX.text, str(row['text']))
        self.add_data_property(post_uri, EX.createdAt, row['time'].isoformat(), XSD.dateTime)
        self.add_data_property(post_uri, EX.depth, depth, XSD.integer)
        self.add_data_property(post_uri, EX.childrenCount, int(row['children_count']), XSD.integer)
        self.add_data_property(post_uri, EX.timeSinceSource, float(row['time_since_source']), XSD.float)
        
        # === V2 EXTENSION: Basic post properties ===
        self.add_data_property(post_uri, EX.postId, str(post_id), XSD.string)
        self.add_data_property(post_uri, EX.isSourcePost, bool(depth == 0 or is_source), XSD.boolean)
        self.add_data_property(post_uri, EX.isReply, bool(depth > 0 and not is_source), XSD.boolean)
        self.add_data_property(post_uri, EX.textLength, len(str(row['text'])), XSD.integer)
        
        # === V2 EXTENSION: Propagation-derived features ===
        if 'position_in_thread' in df_columns:
            pos_val = row['position_in_thread']
            if pd.notna(pos_val):
                self.add_data_property(post_uri, EX.positionInThread, int(pos_val), XSD.integer)
        else:
            logger.debug(f"Column 'position_in_thread' not found, skipping for post {post_id}")
        
        # === V2 EXTENSION: Graph centrality features ===
        graph_columns_float = [
            ('pagerank_score', EX.pagerankScore, XSD.float),
            ('betweenness_centrality', EX.betweennessScore, XSD.float),
            ('closeness_centrality', EX.closenessScore, XSD.float),
        ]
        for col_name, prop_uri, dtype in graph_columns_float:
            if col_name in df_columns:
                val = row[col_name]
                if pd.notna(val):
                    self._safe_add_data_property(post_uri, prop_uri, float(val), dtype)
            else:
                logger.debug(f"Column '{col_name}' not found, skipping for post {post_id}")
        
        graph_columns_int = [
            ('node_in_degree', EX.nodeInDegree, XSD.integer),
            ('node_out_degree', EX.nodeOutDegree, XSD.integer),
            ('subtree_reply_count', EX.subtreeReplyCount, XSD.integer),
            ('sibling_count', EX.siblingCount, XSD.integer),
        ]
        for col_name, prop_uri, dtype in graph_columns_int:
            if col_name in df_columns:
                val = row[col_name]
                if pd.notna(val):
                    self._safe_add_data_property(post_uri, prop_uri, int(val), dtype)
            else:
                logger.debug(f"Column '{col_name}' not found, skipping for post {post_id}")
        
        # Handle repliesTo relationship with validation
        reply_to = row['reply_to']
        
        # Enforce consistency between depth and repliesTo
        if depth == 0:
            # Root post: repliesTo MUST be None
            if pd.notna(reply_to):
                logger.warning(f"Inconsistent data: post {post_id} has depth=0 but repliesTo={reply_to}. Removing incorrect repliesTo edge.")
            # No repliesTo edge should be created for depth=0
        else:
            # Non-root post: repliesTo MUST exist
            if pd.isna(reply_to):
                logger.error(f"Inconsistent data: post {post_id} has depth={depth} but no repliesTo.")
            else:
                # Parse reply_to safely: handle both string scientific notation and direct values
                try:
                    # Convert to float first to handle scientific notation strings, then to int
                    parent_id = int(float(reply_to))
                except (ValueError, TypeError):
                    logger.error(f"Cannot parse reply_to value '{reply_to}' for post {post_id}")
                    return
                
                if parent_id not in self.valid_post_ids:
                    # Target post does NOT exist
                    logger.warning(f"Broken repliesTo relationship: post {post_id} references non-existent parent post {parent_id}")
                else:
                    # Validate structural consistency using post metadata
                    current_thread_id = int(row['thread_id'])
                    parent_metadata = self.post_metadata.get(parent_id)
                    
                    if parent_metadata is None:
                        logger.error(f"Missing metadata for parent post {parent_id}")
                        return
                    
                    parent_thread_id, parent_depth = parent_metadata
                    
                    # Check 1: Parent must be in the same thread
                    if parent_thread_id != current_thread_id:
                        logger.warning(
                            f"Cross-thread repliesTo violation: post {post_id} (thread {current_thread_id}) "
                            f"references parent {parent_id} (thread {parent_thread_id}). Skipping edge."
                        )
                        return
                    
                    # Check 2: Parent depth must be strictly less than child depth
                    if parent_depth >= depth:
                        logger.warning(
                            f"Depth violation: post {post_id} (depth {depth}) references parent {parent_id} "
                            f"(depth {parent_depth}). Skipping edge."
                        )
                        return
                    
                    # All validations passed, create the relationship
                    reply_to_uri = self.create_uri('post', parent_id)
                    self.add_object_property(post_uri, EX.repliesTo, reply_to_uri)
    
    def process_user(self, user_id: int):
        """Process a user and add its triples to the graph (v2 extended)."""
        if user_id in self.created_users:
            return
        
        user_uri = self.create_uri('user', user_id)
        self.add_class_instance(user_uri, EX.User)
        self.created_users.add(user_id)
        
        # === V2 EXTENSION: User data properties ===
        self.add_data_property(user_uri, EX.userId, str(user_id), XSD.string)
        
        # Add precomputed aggregates if available
        if user_id in self.user_aggregates:
            agg = self.user_aggregates[user_id]
            
            self.add_data_property(user_uri, EX.userPostCount, agg['post_count'], XSD.integer)
            self.add_data_property(user_uri, EX.userThreadCount, agg['thread_count'], XSD.integer)
            self._safe_add_data_property(user_uri, EX.userAvgDepth, agg['avg_depth'], XSD.float)
            self._safe_add_data_property(user_uri, EX.userPriorRumorRatio, agg['prior_rumor_ratio'], XSD.float)
            
            # sourceNetworkSize and sourceUserCredibility are per-thread features
            # Assign them from the source user perspective (if available)
            # These may be set later during thread processing
    
    def process_event(self, event_id: str):
        """Process an event and add its triples to the graph (v2 extended)."""
        if event_id in self.created_events:
            return
        
        event_uri = self.create_uri('event', event_id)
        self.add_class_instance(event_uri, EX.Event)
        self.created_events.add(event_id)
        
        # === V2 EXTENSION: Event data properties ===
        self.add_data_property(event_uri, EX.eventId, str(event_id), XSD.string)
        
        # Add hardcoded event info
        if event_id in EVENT_INFO:
            info = EVENT_INFO[event_id]
            self.add_data_property(event_uri, EX.eventName, info["name"], XSD.string)
            self.add_data_property(event_uri, EX.eventType, info["type"], XSD.string)
        else:
            logger.warning(f"Unknown event ID '{event_id}', skipping eventName and eventType")
    
    def process_thread(self, thread_id: int, thread_data: pd.DataFrame, df_columns: set):
        """Process a conversation thread and add its triples to the graph (v2 extended)."""
        if thread_id in self.created_threads:
            return
        
        thread_uri = self.create_uri('thread', thread_id)
        self.add_class_instance(thread_uri, EX.ConversationThread)
        self.created_threads.add(thread_id)
        
        # Add thread-level properties (using the first row as representative)
        first_row = thread_data.iloc[0]
        
        # v1 properties
        self.add_data_property(thread_uri, EX.threadSize, int(first_row['thread_size']), XSD.integer)
        self.add_data_property(thread_uri, EX.maxDepth, int(first_row['max_depth']), XSD.integer)
        self.add_data_property(thread_uri, EX.replySpeed, float(first_row['reply_speed_per_hour']), XSD.float)
        
        # Add veracity label (NonRumor and Rumor are individuals per ontology)
        label_value = int(first_row['label'])
        if label_value == 0:
            label_uri = EX.NonRumor
        else:
            label_uri = EX.Rumor
        
        self.add_object_property(thread_uri, EX.hasVeracity, label_uri)
        
        # === V2 EXTENSION: Thread data properties ===
        self.add_data_property(thread_uri, EX.threadId, str(thread_id), XSD.string)
        self.add_data_property(thread_uri, EX.rumorLabel, label_value, XSD.integer)
        
        # === V2 EXTENSION: belongsToEvent ===
        if thread_id in self.thread_to_event:
            event_id = self.thread_to_event[thread_id]
            event_uri = self.create_uri('event', event_id)
            self.add_object_property(thread_uri, EX.belongsToEvent, event_uri)
        
        # === V2 EXTENSION: hasSourcePost ===
        if thread_id in self.thread_to_source_post:
            source_post_id = self.thread_to_source_post[thread_id]
            source_post_uri = self.create_uri('post', source_post_id)
            self.add_object_property(thread_uri, EX.hasSourcePost, source_post_uri)
    
    def _add_participates_in_thread(self, df: pd.DataFrame):
        """Add ex:participatesInThread relationships for all (user, thread) pairs."""
        logger.info("Adding participatesInThread relationships...")
        
        # Get unique (user_id, thread_id) pairs
        pairs = df[['user_id', 'thread_id']].drop_duplicates()
        count = 0
        
        for _, row in pairs.iterrows():
            user_id = int(row['user_id'])
            thread_id = int(row['thread_id'])
            
            user_uri = self.create_uri('user', user_id)
            thread_uri = self.create_uri('thread', thread_id)
            
            # Only add if both entities exist in the graph
            if user_id in self.created_users and thread_id in self.created_threads:
                self.add_object_property(user_uri, EX.participatesInThread, thread_uri)
                count += 1
        
        logger.info(f"Added {count} participatesInThread relationships")
    
    def build_knowledge_graph(self, df: pd.DataFrame):
        """Build the complete knowledge graph from the dataset (v2 extended)."""
        logger.info("Building knowledge graph v2...")
        
        df_columns = set(df.columns)
        
        # Step 0: Precompute aggregates and metadata
        self._precompute_user_aggregates(df)
        self._precompute_thread_metadata(df)
        
        # Step 1: Collect all valid post_ids and their metadata for repliesTo validation
        self.valid_post_ids = set(df['post_id'].unique())
        logger.info(f"Collected {len(self.valid_post_ids)} valid post_ids for repliesTo validation")
        
        # Build post metadata lookup for structural validation
        for _, row in df.iterrows():
            post_id = int(row['post_id'])
            thread_id = int(row['thread_id'])
            depth = int(row['depth'])
            self.post_metadata[post_id] = (thread_id, depth)
        logger.info(f"Built metadata for {len(self.post_metadata)} posts for structural validation")
        
        # Step 2: Group by thread for efficient processing
        thread_groups = df.groupby('thread_id')
        
        for thread_id, thread_data in thread_groups:
            # Process thread first
            self.process_thread(int(thread_id), thread_data, df_columns)
            
            # Process each post in the thread
            for _, post_row in thread_data.iterrows():
                self.process_post(post_row, df_columns)
                
                # Process associated entities
                self.process_user(int(post_row['user_id']))
                self.process_event(post_row['event_id'])
        
        # Step 3: Add participatesInThread relationships (v2)
        self._add_participates_in_thread(df)
        
        # Step 4: Post-process: assign sourceNetworkSize and sourceUserCredibility
        # to source users (the user who started each thread)
        self._add_source_user_properties(df)
        
        logger.info(f"Knowledge graph v2 built with {len(self.graph)} triples")
        logger.info(f"Entities created: {len(self.created_posts)} posts, {len(self.created_users)} users, "
                   f"{len(self.created_events)} events, {len(self.created_threads)} threads")
    
    def _add_source_user_properties(self, df: pd.DataFrame):
        """Assign sourceNetworkSize and sourceUserCredibility to source users."""
        logger.info("Adding source user properties...")
        
        for thread_id, source_post_id in self.thread_to_source_post.items():
            # Find the source post row
            source_rows = df[df['post_id'] == source_post_id]
            if len(source_rows) == 0:
                continue
            
            source_row = source_rows.iloc[0]
            source_user_id = int(source_row['user_id'])
            source_user_uri = self.create_uri('user', source_user_id)
            
            df_columns = set(df.columns)
            
            # source_network_size
            if 'source_network_size' in df_columns:
                val = source_row['source_network_size']
                if pd.notna(val):
                    self._safe_add_data_property(source_user_uri, EX.sourceNetworkSize, int(val), XSD.integer)
            elif source_user_id in self.user_aggregates:
                self.add_data_property(source_user_uri, EX.sourceNetworkSize, 
                                      self.user_aggregates[source_user_id]['post_count'], XSD.integer)
            
            # source_user_credibility
            if 'source_user_credibility' in df_columns:
                val = source_row['source_user_credibility']
                if pd.notna(val):
                    self._safe_add_data_property(source_user_uri, EX.sourceUserCredibility, float(val), XSD.float)
            elif source_user_id in self.user_aggregates:
                credibility = 1.0 - self.user_aggregates[source_user_id]['prior_rumor_ratio']
                self._safe_add_data_property(source_user_uri, EX.sourceUserCredibility, credibility, XSD.float)
    
    def _collect_reply_edges(self) -> list:
        """Collect all repliesTo edges from the graph as (source_id, target_id, subject_uri, object_uri) tuples."""
        edges = []
        for subject, predicate, obj in self.graph.triples((None, EX.repliesTo, None)):
            source_id = self.extract_post_id_from_uri(subject)
            target_id = self.extract_post_id_from_uri(obj)
            if source_id is not None and target_id is not None:
                edges.append((source_id, target_id, subject, obj))
        return edges
    
    def _detect_cycle_edges(self, edges: list) -> list:
        """Detect cycles and return list of edges to remove (without mutating the graph)."""
        # Build adjacency list
        reply_graph = {}
        for source_id, target_id, _, _ in edges:
            if source_id not in reply_graph:
                reply_graph[source_id] = []
            reply_graph[source_id].append(target_id)
        
        logger.info(f"Analyzing {len(edges)} reply relationships for cycles")
        
        visited = set()
        rec_stack = set()
        edges_to_remove = set()  # Store (source_id, target_id) tuples to remove
        
        def dfs_detect_cycle(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Check for self-loops
            if node_id in reply_graph and node_id in reply_graph[node_id]:
                logger.warning(f"Self-loop detected: post {node_id} → itself")
                edges_to_remove.add((node_id, node_id))
                return True
            
            cycle_found = False
            if node_id in reply_graph:
                for neighbor_id in list(reply_graph[node_id]):  # Use list copy
                    if neighbor_id not in visited:
                        if dfs_detect_cycle(neighbor_id):
                            cycle_found = True
                    elif neighbor_id in rec_stack:
                        # Cycle detected - mark edge for removal
                        logger.warning(f"Cycle detected involving edge: {node_id} → {neighbor_id}")
                        edges_to_remove.add((node_id, neighbor_id))
                        cycle_found = True
            
            rec_stack.remove(node_id)
            return cycle_found
        
        # Run cycle detection on all nodes
        for node_id in reply_graph:
            if node_id not in visited:
                dfs_detect_cycle(node_id)
        
        return list(edges_to_remove)
    
    def detect_and_remove_cycles(self) -> bool:
        """Detect and remove cycles in the reply tree graph using DFS."""
        logger.info("Detecting and removing cycles in reply tree...")
        
        # Step 1: Collect all reply edges
        edges = self._collect_reply_edges()
        
        # Step 2: Detect cycle edges (without mutating graph)
        edges_to_remove = self._detect_cycle_edges(edges)
        
        # Step 3: Remove only the problematic edges from the graph
        if edges_to_remove:
            logger.warning(f"Found and removing {len(edges_to_remove)} cycle edges")
            for source_id, target_id in edges_to_remove:
                source_uri = self.create_uri('post', source_id)
                target_uri = self.create_uri('post', target_id)
                self.graph.remove((source_uri, EX.repliesTo, target_uri))
                logger.info(f"Removed cycle edge: {source_id} → {target_id}")
        else:
            logger.info("No cycles detected in reply tree")
        
        return len(edges_to_remove) == 0
    
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
        
        logger.info("✅ Knowledge graph v2 validation passed")
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
    
    def _get_triple_count_by_subject_type(self) -> dict:
        """Count triples grouped by the type of the subject entity."""
        counts = {
            'Post': 0,
            'User': 0,
            'Event': 0,
            'Thread': 0,
            'Other': 0
        }
        
        for subject, predicate, obj in self.graph:
            uri_str = str(subject)
            if '/post/' in uri_str:
                counts['Post'] += 1
            elif '/user/' in uri_str:
                counts['User'] += 1
            elif '/event/' in uri_str:
                counts['Event'] += 1
            elif '/thread/' in uri_str:
                counts['Thread'] += 1
            else:
                counts['Other'] += 1
        
        return counts
    
    def print_statistics(self):
        """Print comprehensive statistics about the built KG v2."""
        print(f"\n=== Knowledge Graph v2 Statistics ===")
        print(f"  File: data/processed/pheme_kg_v2.ttl")
        print(f"  Total triples: {len(self.graph)}")
        
        type_counts = self._get_triple_count_by_subject_type()
        print(f"  Triples by subject type:")
        print(f"    - Post:   {type_counts['Post']}")
        print(f"    - User:   {type_counts['User']}")
        print(f"    - Event:  {type_counts['Event']}")
        print(f"    - Thread: {type_counts['Thread']}")
        print(f"    - Other:  {type_counts['Other']}")
        
        print(f"  Entities created:")
        print(f"    - Posts:  {len(self.created_posts)}")
        print(f"    - Users:  {len(self.created_users)}")
        print(f"    - Events: {len(self.created_events)}")
        print(f"    - Threads: {len(self.created_threads)}")
        
        # Count SourcePost and ReplyPost instances
        source_count = len(list(self.graph.subjects(RDF.type, EX.SourcePost)))
        reply_count = len(list(self.graph.subjects(RDF.type, EX.ReplyPost)))
        print(f"  Post subtypes:")
        print(f"    - SourcePost: {source_count}")
        print(f"    - ReplyPost:  {reply_count}")
        
        # Count reply relationships
        reply_rels = len(list(self.graph.triples((None, EX.repliesTo, None))))
        print(f"  Reply relationships: {reply_rels}")
        
        # Count participatesInThread
        part_rels = len(list(self.graph.triples((None, EX.participatesInThread, None))))
        print(f"  participatesInThread: {part_rels}")
        
        print(f"  [OK] Knowledge graph v2 saved successfully")
    
    def build_complete_kg(self, input_file: str, output_file: str):
        """Complete pipeline: load data, build KG, validate, and save."""
        logger.info("Starting complete knowledge graph v2 construction pipeline...")
        
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
        
        # Step 6: Print statistics
        self.print_statistics()
        
        logger.info("✅ Knowledge graph v2 construction completed successfully!")


def main():
    """Main function to run the knowledge graph v2 construction."""
    builder = KnowledgeGraphBuilderV2()
    
    # Prefer pheme_features_with_graph.csv, fallback to pheme_features.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file_with_graph = os.path.join(base_dir, "data", "processed", "pheme_features_with_graph.csv")
    input_file_basic = os.path.join(base_dir, "data", "processed", "pheme_features.csv")
    
    if os.path.exists(input_file_with_graph):
        input_file = input_file_with_graph
        logger.info("Using pheme_features_with_graph.csv (includes graph features)")
    else:
        input_file = input_file_basic
        logger.info("pheme_features_with_graph.csv not found, falling back to pheme_features.csv")
    
    output_file = os.path.join(base_dir, "data", "processed", "pheme_kg_v2.ttl")
    
    try:
        builder.build_complete_kg(input_file, output_file)
    except Exception as e:
        logger.error(f"Error during knowledge graph v2 construction: {str(e)}")
        raise


if __name__ == "__main__":
    main()
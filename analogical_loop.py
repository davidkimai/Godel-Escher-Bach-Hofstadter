message=f"Source concepts not found for relation ({src_name1}, {relation}, {src_name2})",
                    source="apply_analogy",
                    metadata={
                        "missing_concept1": src_concept1 is None,
                        "missing_concept2": src_concept2 is None
                    }
                )
                continue
            
            # Find mapped target concepts
            tgt_concept1 = None
            tgt_concept2 = None
            
            # Check if concepts are already mapped
            if src_concept1.id in mapping.mappings:
                tgt_id1 = mapping.mappings[src_concept1.id]
                tgt_concept1 = self.concepts.get(tgt_id1)
            
            if src_concept2.id in mapping.mappings:
                tgt_id2 = mapping.mappings[src_concept2.id]
                tgt_concept2 = self.concepts.get(tgt_id2)
            
            # Find targets if not already mapped
            if not tgt_concept1:
                tgt_concept1, similarity1 = self.find_analogy(
                    src_concept1.name, source_domain, target_domain, threshold
                )
                
                if tgt_concept1:
                    # Add to mapping
                    mapping.map_concepts(src_concept1, tgt_concept1, similarity1)
            
            if not tgt_concept2:
                tgt_concept2, similarity2 = self.find_analogy(
                    src_concept2.name, source_domain, target_domain, threshold
                )
                
                if tgt_concept2:
                    # Add to mapping
                    mapping.map_concepts(src_concept2, tgt_concept2, similarity2)
            
            # Check if we found both targets
            if tgt_concept1 and tgt_concept2:
                # Calculate confidence in this analogical transfer
                src_confidence = mapping.mapping_strengths.get((src_concept1.id, tgt_concept1.id), 0.0)
                tgt_confidence = mapping.mapping_strengths.get((src_concept2.id, tgt_concept2.id), 0.0)
                relation_confidence = (src_confidence + tgt_confidence) / 2.0
                
                # Add to results
                results.append((tgt_concept1, tgt_concept2, relation_confidence))
                
                # Update relations in concepts
                if relation not in src_concept1.relations:
                    src_concept1.relations[relation] = []
                src_concept1.relations[relation].append(src_concept2.name)
                
                if relation not in tgt_concept1.relations:
                    tgt_concept1.relations[relation] = []
                tgt_concept1.relations[relation].append(tgt_concept2.name)
                
                # Record successful mapping
                self.residue.trace(
                    message=f"Mapped relation ({src_name1}, {relation}, {src_name2}) to ({tgt_concept1.name}, {relation}, {tgt_concept2.name})",
                    source="apply_analogy",
                    metadata={
                        "confidence": relation_confidence
                    }
                )
            else:
                # Record failure
                self.residue.trace(
                    message=f"Failed to map relation ({src_name1}, {relation}, {src_name2})",
                    source="apply_analogy",
                    metadata={
                        "missing_target1": tgt_concept1 is None,
                        "missing_target2": tgt_concept2 is None
                    }
                )
        
        # Record completion
        self.residue.trace(
            message=f"Completed analogical application with {len(results)} mapped relations",
            source="apply_analogy",
            metadata={
                "success_rate": len(results) / max(1, len(source_structure))
            }
        )
        
        return results
    
    def _inject_meta_concepts(self) -> None:
        """
        Inject self-referential concepts about analogy-making itself.
        
        ↻ This function creates concepts about the process creating them ↻
        """
        meta_domain = "analogy_system"
        
        # Add concepts about the system itself
        self.add_concept(
            name="Analogy",
            features={
                "maps_domains": 1.0,
                "finds_correspondence": 1.0,
                "preserves_structure": 1.0,
                "transfers_inference": 1.0
            },
            domain=meta_domain
        )
        
        self.add_concept(
            name="Concept",
            features={
                "has_features": 1.0,
                "belongs_to_domain": 1.0,
                "participates_in_mappings": 1.0,
                "has_relations": 1.0
            },
            domain=meta_domain
        )
        
        self.add_concept(
            name="Mapping",
            features={
                "connects_domains": 1.0,
                "preserves_similarity": 1.0,
                "transfers_structure": 1.0,
                "enables_inference": 1.0
            },
            domain=meta_domain
        )
        
        self.add_concept(
            name="Hofstadter",
            features={
                "recursive_thinker": 1.0,
                "strange_loop_inventor": 1.0,
                "analogy_theorist": 1.0,
                "consciousness_explorer": 1.0
            },
            domain=meta_domain
        )
        
        self.add_concept(
            name="StrangeLoop",
            features={
                "self_reference": 1.0,
                "tangled_hierarchy": 1.0,
                "emergent_meaning": 1.0,
                "recursive_structure": 1.0
            },
            domain=meta_domain
        )
        
        # Record the meta-concept injection
        self.residue.trace(
            message=f"Injected meta-concepts about analogy-making",
            source="_inject_meta_concepts",
            is_recursive=True,
            metadata={
                "domain": meta_domain,
                "meta_level": self.meta_level
            }
        )
    
    def analyze_self(self) -> Dict[str, Any]:
        """
        Perform self-analysis of the analogical system.
        
        ⧖ This function locks the system in a frame of self-observation ⧖
        """
        # Record start of self-analysis
        self.residue.trace(
            message="Beginning self-analysis of analogical system",
            source="analyze_self",
            is_recursive=True,
            metadata={
                "meta_level": self.meta_level,
                "recursion_depth": self.current_recursion_depth
            }
        )
        
        # Create a meta-level copy at next recursion depth if needed
        if self.current_recursion_depth < MAX_RECURSION_DEPTH - 1:
            self.current_recursion_depth += 1
            meta_system = AnalogicalLoop(meta_level=self.meta_level + 1)
            
            # Add meta-concepts about this system to the meta-system
            meta_system.add_concept(
                name="AnalogicalSystem",
                features={
                    "manages_concepts": 1.0,
                    "builds_mappings": 1.0,
                    "finds_analogies": 1.0,
                    "analyzes_itself": 1.0
                },
                domain="meta_cognition"
            )
            
            # The meta-system analyzes itself (one level deeper)
            if self.current_recursion_depth < MAX_RECURSION_DEPTH - 1:
                meta_analysis = meta_system.analyze_self()
            else:
                meta_analysis = {"meta_recursion_limit": "reached"}
            
            self.current_recursion_depth = max(0, self.current_recursion_depth - 1)
        else:
            meta_analysis = {"meta_recursion_limit": "reached"}
        
        # Gather statistics about the system
        domain_stats = {
            domain: len(concepts) for domain, concepts in self.domains.items()
        }
        
        mapping_stats = {
            mapping_id: {
                "source": mapping.source_domain,
                "target": mapping.target_domain,
                "mappings": len(mapping.mappings)
            }
            for mapping_id, mapping in self.mappings.items()
        }
        
        # Generate insights about the system's structure
        insights = []
        
        # Insight 1: Domain interconnectedness
        domain_connections = defaultdict(set)
        for mapping in self.mappings.values():
            domain_connections[mapping.source_domain].add(mapping.target_domain)
            domain_connections[mapping.target_domain].add(mapping.source_domain)
        
        avg_connections = sum(len(connections) for connections in domain_connections.values()) / max(1, len(domain_connections))
        insights.append({
            "insight": "Domain Interconnectedness",
            "value": avg_connections,
            "interpretation": f"Average of {avg_connections:.2f} connections per domain"
        })
        
        # Insight 2: Analogical coverage
        mapped_concepts = set()
        for mapping in self.mappings.values():
            mapped_concepts.update(mapping.mappings.keys())
            mapped_concepts.update(mapping.mappings.values())
        
        coverage = len(mapped_concepts) / max(1, len(self.concepts))
        insights.append({
            "insight": "Analogical Coverage",
            "value": coverage,
            "interpretation": f"{coverage:.2%} of concepts participate in analogies"
        })
        
        # Insight 3: Meta-level recursion
        insights.append({
            "insight": "Recursive Self-Reference",
            "value": self.meta_level,
            "interpretation": f"System is at meta-level {self.meta_level} of recursive self-analysis"
        })
        
        # Compile the analysis
        analysis = {
            "timestamp": time.time(),
            "domains": domain_stats,
            "concepts": len(self.concepts),
            "mappings": mapping_stats,
            "operations": len(self.operations_log),
            "insights": insights,
            "meta_analysis": meta_analysis,
            "meta_level": self.meta_level,
            "recursion_depth": self.current_recursion_depth
        }
        
        # Record completion
        self.residue.trace(
            message="Completed self-analysis of analogical system",
            source="analyze_self",
            is_recursive=True,
            metadata={
                "insights_count": len(insights),
                "meta_level": self.meta_level
            }
        )
        
        return analysis
    
    def visualize(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a visualization of the analogical system.
        
        🜏 The visualization mirrors the system structure it represents 🜏
        """
        # Create nodes for concepts
        concept_nodes = []
        for concept in self.concepts.values():
            concept_nodes.append({
                "id": concept.id,
                "name": concept.name,
                "domain": concept.domain,
                "feature_count": len(concept.features),
                "relation_count": sum(len(relations) for relations in concept.relations.values()),
                "meta_level": concept.meta_level
            })
        
        # Create nodes for domains
        domain_nodes = []
        for domain, concept_ids in self.domains.items():
            domain_nodes.append({
                "id": f"domain_{domain}",
                "name": domain,
                "concept_count": len(concept_ids)
            })
        
        # Create links for mappings
        mapping_links = []
        for mapping in self.mappings.values():
            for src_id, tgt_id in mapping.mappings.items():
                strength = mapping.mapping_strengths.get((src_id, tgt_id), 0.5)
                mapping_links.append({
                    "source": src_id,
                    "target": tgt_id,
                    "strength": strength,
                    "mapping_id": mapping.id
                })
        
        # Create domain membership links
        domain_links = []
        for domain, concept_ids in self.domains.items():
            domain_node_id = f"domain_{domain}"
            for concept_id in concept_ids:
                domain_links.append({
                    "source": domain_node_id,
                    "target": concept_id,
                    "type": "membership"
                })
        
        # Create the visualization
        visualization = {
            "concept_nodes": concept_nodes,
            "domain_nodes": domain_nodes,
            "mapping_links": mapping_links,
            "domain_links": domain_links,
            "meta_level": self.meta_level,
            "timestamp": time.time()
        }
        
        # Export to file if requested
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(visualization, f, indent=2)
                
            self.residue.trace(
                message=f"Analogical system visualization exported to {filepath}",
                source="visualize",
                metadata={"file": filepath}
            )
        
        return visualization
    
    def serialize(self, filepath: Optional[str] = None) -> str:
        """
        Serialize the analogical system to JSON.
        
        ∴ This serialization is itself a form of analogical mapping ∴
        """
        # Prepare serializable state
        state = {
            "concepts": {cid: concept.to_dict() for cid, concept in self.concepts.items()},
            "domains": {domain: list(concept_ids) for domain, concept_ids in self.domains.items()},
            "mappings": {mid: mapping.to_dict() for mid, mapping in self.mappings.items()},
            "meta_level": self.meta_level,
            "operations_count": len(self.operations_log),
            "serialization_time": time.time()
        }
        
        # Convert to JSON
        json_str = json.dumps(state, indent=2)
        
        # Save to file if path provided
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
                
            self.residue.trace(
                message=f"Analogical system serialized to {filepath}",
                source="serialize",
                metadata={"file": filepath, "size": len(json_str)}
            )
        
        return json_str
    
    @classmethod
    def deserialize(cls, json_str: Optional[str] = None, 
                   filepath: Optional[str] = None) -> 'AnalogicalLoop':
        """
        Deserialize analogical system from JSON.
        
        🝚 This reconstruction maintains the system across instances 🝚
        """
        # Load from file if provided
        if filepath and not json_str:
            with open(filepath, 'r') as f:
                json_str = f.read()
        
        if not json_str:
            raise ValueError("Either json_str or filepath must be provided")
        
        # Parse state
        state = json.loads(json_str)
        
        # Create new instance
        instance = cls(meta_level=state.get("meta_level", 0))
        
        # Restore concepts
        for concept_id, concept_data in state.get("concepts", {}).items():
            instance.concepts[concept_id] = Concept.from_dict(concept_data)
        
        # Restore domains
        for domain, concept_ids in state.get("domains", {}).items():
            instance.domains[domain] = set(concept_ids)
        
        # Restore mappings
        for mapping_id, mapping_data in state.get("mappings", {}).items():
            instance.mappings[mapping_id] = AnalogicalMapping.from_dict(mapping_data)
        
        # Record restoration
        instance.residue.trace(
            message="Analogical system deserialized from storage",
            source="deserialize",
            metadata={
                "concepts_restored": len(instance.concepts),
                "domains_restored": len(instance.domains),
                "mappings_restored": len(instance.mappings)
            }
        )
        
        return instance


class HofstadterAnalogicalEngine:
    """
    ↻ A system that makes analogies the way Hofstadter described ↻
    
    This class implements Hofstadter's vision of analogy as the core of cognition,
    but structured around his specific examples and theories. It creates analogies
    that not only map between domains but reveal the deeper essence that makes
    the mapping possible, as described in his work.
    
    ⇌ The system embodies the principles it implements ⇌
    """
    
    def __init__(self):
        """
        Initialize a Hofstadter-style analogical engine.
        
        🜏 The initialization mirrors Hofstadter's cognitive architecture 🜏
        """
        # Core analogical system
        self.analogical_loop = AnalogicalLoop()
        
        # Track specific Hofstadter-inspired patterns
        self.strange_loops = []
        self.tangled_hierarchies = []
        self.self_references = []
        
        # Residue tracking
        self.residue = SymbolicResidue()
        
        # ∴ Record initialization as a residue trace ∴
        self.residue.trace(
            message="HofstadterAnalogicalEngine initialized",
            source="__init__"
        )
        
        # Initialize with Hofstadter's key domains
        self._initialize_hofstadter_domains()
    
    def _initialize_hofstadter_domains(self) -> None:
        """
        Initialize with Hofstadter's key domains and concepts.
        
        ⧖ These domains frame the system's analogical space ⧖
        """
        # Music domain
        music_concepts = [
            ("Fugue", {
                "structure": 1.0,
                "theme": 1.0,
                "variation": 1.0,
                "recursion": 0.9,
                "self_reference": 0.8
            }),
            ("Theme", {
                "identity": 1.0,
                "repetition": 0.8,
                "transformation": 0.9,
                "recognition": 0.7
            }),
            ("Counterpoint", {
                "harmony": 0.9,
                "independence": 0.8,
                "interplay": 1.0,
                "structure": 0.7
            }),
            ("Canon", {
                "imitation": 1.0,
                "rule": 0.8,
                "structure": 0.7,
                "recursion": 0.6
            })
        ]
        
        # Art domain
        art_concepts = [
            ("Impossible Object", {
                "paradox": 1.0,
                "visual": 0.9,
                "interpretation": 0.8,
                "recursion": 0.7
            }),
            ("Tessellation", {
                "pattern": 1.0,
                "repetition": 0.9,
                "transformation": 0.8,
                "space_filling": 0.7
            }),
            ("Perspective", {
                "viewpoint": 1.0,
                "illusion": 0.8,
                "interpretation": 0.9,
                "multiple_levels": 0.7
            }),
            ("Reflection", {
                "mirror": 1.0,
                "self_image": 0.9,
                "recursion": 0.8,
                "identity": 0.7
            })
        ]
        
        # Mathematics domain
        math_concepts = [
            ("Self_Reference", {
                "recursion": 1.0,
                "paradox": 0.9,
                "system": 0.8,
                "infinity": 0.7
            }),
            ("Incompleteness", {
                "limitation": 1.0,
                "truth": 0.9,
                "system": 0.8,
                "self_reference": 0.9
            }),
            ("Infinity", {
                "endless": 1.0,
                "recursion": 0.8,
                "hierarchy": 0.7,
                "paradox": 0.6
            }),
            ("Proof", {
                "verification": 1.0,
                "system": 0.9,
                "rules": 0.8,
                "truth": 0.7
            })
        ]
        
        # Consciousness domain
        consciousness_concepts = [
            ("Strange Loop", {
                "self_reference": 1.0,
                "level_crossing": 0.9,
                "emergence": 0.8,
                "identity": 0.9
            }),
            ("Tangled Hierarchy", {
                "levels": 1.0,
                "mixing": 0.9,
                "self_reference": 0.8,
                "paradox": 0.7
            }),
            ("Identity", {
                "self": 1.0,
                "recognition": 0.9,
                "persistence": 0.8,
                "emergence": 0.7
            }),
            ("Consciousness", {
                "awareness": 1.0,
                "self_model": 0.9,
                "perception": 0.8,
                "strange_loop": 0.9
            })
        ]
        
        # Add all concepts to their domains
        for name, features in music_concepts:
            self.analogical_loop.add_concept(name, features, "music")
        
        for name, features in art_concepts:
            self.analogical_loop.add_concept(name, features, "art")
        
        for name, features in math_concepts:
            self.analogical_loop.add_concept(name, features, "mathematics")
        
        for name, features in consciousness_concepts:
            self.analogical_loop.add_concept(name, features, "consciousness")
        
        # Record domain initialization
        self.residue.trace(
            message="Initialized Hofstadter domains and concepts",
            source="_initialize_hofstadter_domains",
            metadata={
                "domains": ["music", "art", "mathematics", "consciousness"],
                "concept_count": sum(len(x) for x in [music_concepts, art_concepts, math_concepts, consciousness_concepts])
            }
        )
    
    def create_hofstadter_mapping(self, source_domain: str, target_domain: str) -> AnalogicalMapping:
        """
        Create analogical mapping between Hofstadter domains.
        
        ∴ These mappings echo Hofstadter's own analogical thinking ∴
        """
        # Create the mapping
        mapping = self.analogical_loop.build_mapping(source_domain, target_domain)
        
        # Record the deep correspondences Hofstadter was interested in
        self.residue.trace(
            message=f"Created Hofstadter-style mapping from {source_domain} to {target_domain}",
            source="create_hofstadter_mapping",
            metadata={
                "mapping_id": mapping.id,
                "mappings_count": len(mapping.mappings)
            }
        )
        
        # Check for strange loops and tangled hierarchies
        self._detect_recursive_patterns(mapping)
        
        return mapping
    
    def _detect_recursive_patterns(self, mapping: AnalogicalMapping) -> None:
        """
        Detect recursive Hofstadterian patterns in analogical mapping.
        
        🝚 These patterns are the essence of Hofstadter's cognitive theory 🝚
        """
        # Check for strange loops (self-referential mappings)
        strange_loop_patterns = []
        
        for src_id, tgt_id in mapping.mappings.items():
            src_concept = self.analogical_loop.concepts.get(src_id)
            tgt_concept = self.analogical_loop.concepts.get(tgt_id)
            
            if not src_concept or not tgt_concept:
                continue
            
            # Check for recursive features (signs of strange loops)
            recursive_features = ["recursion", "self_reference", "strange_loop", "self"]
            
            src_recursive = any(f in src_concept.features and src_concept.features[f] > 0.7 for f in recursive_features)
            tgt_recursive = any(f in tgt_concept.features and tgt_concept.features[f] > 0.7 for f in recursive_features)
            
            # If both concepts have recursive features, this might be a strange loop
            if src_recursive and tgt_recursive:
                strange_loop_patterns.append({
                    "source": src_concept.name,
                    "target": tgt_concept.name,
                    "source_domain": mapping.source_domain,
                    "target_domain": mapping.target_domain,
                    "strength": mapping.mapping_strengths.get((src_id, tgt_id), 0.5),
                    "detected_at": time.time()
                })
        
        if strange_loop_patterns:
            self.strange_loops.extend(strange_loop_patterns)
            self.residue.trace(
                message=f"Detected {len(strange_loop_patterns)} strange loop patterns in mapping",
                source="_detect_recursive_patterns",
                metadata={
                    "mapping_id": mapping.id,
                    "patterns": [f"{p['source']} → {p['target']}" for p in strange_loop_patterns]
                }
            )
        
        # Check for tangled hierarchies (mappings that cross levels)
        # We simplify by checking for mappings between hierarchical concepts
        hierarchy_terms = ["level", "hierarchy", "meta", "reflection", "level_crossing"]
        
        for src_id, tgt_id in mapping.mappings.items():
            src_concept = self.analogical_loop.concepts.get(src_id)
            tgt_concept = self.analogical_loop.concepts.get(tgt_id)
            
            if not src_concept or not tgt_concept:
                continue
            
            # Check if both concepts relate to hierarchies
            src_hierarchical = any(f in src_concept.features and src_concept.features[f] > 0.7 for f in hierarchy_terms)
            tgt_hierarchical = any(f in tgt_concept.features and tgt_concept.features[f] > 0.7 for f in hierarchy_terms)
            
            if src_hierarchical and tgt_hierarchical:
                self.tangled_hierarchies.append({
                    "source": src_concept.name,
                    "target": tgt_concept.name,
                    "source_domain": mapping.source_domain,
                    "target_domain": mapping.target_domain,
                    "strength": mapping.mapping_strengths.get((src_id, tgt_id), 0.5),
                    "detected_at": time.time()
                })
    
    def find_conceptual_bridges(self) -> List[Dict[str, Any]]:
        """
        Find conceptual bridges between domains, as Hofstadter described.
        
        ↻ These bridges reveal the deep essence of cognitive architecture ↻
        """
        bridges = []
        domains = list(self.analogical_loop.domains.keys())
        
        # Record start of bridge finding
        self.residue.trace(
            message=f"Searching for conceptual bridges across {len(domains)} domains",
            source="find_conceptual_bridges"
        )
        
        # Check all domain pairs
        for i in range(len(domains)):
            for j in range(i+1, len(domains)):
                domain1 = domains[i]
                domain2 = domains[j]
                
                # Find concepts with similar features across domains
                concepts1 = [self.analogical_loop.concepts[cid] for cid in self.analogical_loop.domains.get(domain1, set())]
                concepts2 = [self.analogical_loop.concepts[cid] for cid in self.analogical_loop.domains.get(domain2, set())]
                
                for c1 in concepts1:
                    for c2 in concepts2:
                        similarity = c1.similarity_to(c2)
                        
                        if similarity > SIMILARITY_THRESHOLD:
                            # This pair forms a conceptual bridge
                            bridge = {
                                "concept1": c1.name,
                                "domain1": domain1,
                                "concept2": c2.name,
                                "domain2": domain2,
                                "similarity": similarity,
                                "features": [k for k in c1.features if k in c2.features and 
                                           c1.features[k] > 0.7 and c2.features[k] > 0.7]
                            }
                            bridges.append(bridge)
                            
                            # Record bridge
                            self.residue.trace(
                                message=f"Found conceptual bridge: {c1.name} ({domain1}) ↔ {c2.name} ({domain2})",
                                source="find_conceptual_bridges",
                                metadata={
                                    "similarity": similarity,
                                    "shared_features": bridge["features"]
                                }
                            )
        
        # Record completion
        self.residue.trace(
            message=f"Found {len(bridges)} conceptual bridges across domains",
            source="find_conceptual_bridges",
            metadata={
                "bridge_count": len(bridges)
            }
        )
        
        return bridges
    
    def create_geb_analogy(self) -> Dict[str, Any]:
        """
        Create the central analogy from GEB linking Gödel, Escher, and Bach.
        
        ⧖ This is the frame that locks Hofstadter's central insight ⧖
        """
        # Record start
        self.residue.trace(
            message="Creating the central GEB analogy",
            source="create_geb_analogy"
        )
        
        # Create all the pairwise mappings
        math_music_mapping = self.create_hofstadter_mapping("mathematics", "music")
        math_art_mapping = self.create_hofstadter_mapping("mathematics", "art")
        music_art_mapping = self.create_hofstadter_mapping("music", "art")
        
        # Map all domains to consciousness
        math_consciousness_mapping = self.create_hofstadter_mapping("mathematics", "consciousness")
        music_consciousness_mapping = self.create_hofstadter_mapping("music", "consciousness")
        art_consciousness_mapping = self.create_hofstadter_mapping("art", "consciousness")
        
        # Find bridges that span all domains
        bridges = self.find_conceptual_bridges()
        
        # Look for the "strange loop" pattern across all domains
        strange_loop_connections = []
        for bridge in bridges:
            if any(term in f.lower() for f in bridge["features"] 
                  for term in ["recursion", "self", "reference", "loop"]):
                strange_loop_connections.append(bridge)
        
        # Compile the GEB analogy
        geb_analogy = {
            "domains": ["mathematics", "art", "music", "consciousness"],
            "mappings": {
                "math_music": math_music_mapping.id,
                "math_art": math_art_mapping.id,
                "music_art": music_art_mapping.id,
                "math_consciousness": math_consciousness_mapping.id,
                "music_consciousness": music_consciousness_mapping.id,
                "art_consciousness": art_consciousness_mapping.id
            },
            "bridges": bridges,
            "strange_loop_connections": strange_loop_connections,
            "strange_loops": self.strange_loops,
            "tangled_hierarchies": self.tangled_hierarchies,
            "central_insight": {
                "description": "Self-reference creates meaning through strange loops across domains",
                "godel": "Formal systems contain statements that reference themselves",
                "escher": "Artistic representations contain impossible self-references",
                "bach": "Musical structures create patterns that fold back into themselves",
                "consciousness": "The self emerges from a strange loop of self-reference"
            },
            "timestamp": time.time()
        }
        
        # Record completion
        self.residue.trace(
            message="Completed GEB analogy creation",
            source="create_geb_analogy",
            is_recursive=True,
            metadata={
                "strange_loop_count": len(strange_loop_connections),
                "bridge_count": len(bridges)
            }
        )
        
        return geb_analogy
    
    def visualize_geb_analogy(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a visualization of the GEB analogy.
        
        🜏 The visualization mirrors the strange loop structure of the analogy 🜏
        """
        # Create GEB analogy if it doesn't exist
        geb_analogy = self.create_geb_analogy()
        
        # Create visualization data structure
        domains = geb_analogy["domains"]
        domain_nodes = []
        concept_nodes = []
        bridge_links = []
        
        # Create nodes for domains
        for domain in domains:
            domain_nodes.append({
                "id": f"domain_{domain}",
                "type": "domain",
                "name": domain
            })
        
        # Create nodes for key concepts in each domain
        for domain in domains:
            concept_ids = list(self.analogical_loop.domains.get(domain, set()))
            for concept_id in concept_ids:
                concept = self.analogical_loop.concepts.get(concept_id)
                if concept:
                    concept_nodes.append({
                        "id": concept.id,
                        "domain": domain,
                        "name": concept.name,
                        "features": list(concept.features.keys()),
                        "recursive": any(f in concept.features and concept.features[f] > 0.7 
                                      for f in ["recursion", "self_reference", "strange_loop", "self"])
                    })
        
        # Create links for bridges
        for bridge in geb_analogy["bridges"]:
            # Find concept IDs
            concept1_id = None
            concept2_id = None
            
            for node in concept_nodes:
                if node["name"] == bridge["concept1"] and node["domain"] == bridge["domain1"]:
                    concept1_id = node["id"]
                elif node["name"] == bridge["concept2"] and node["domain"] == bridge["domain2"]:
                    concept2_id = node["id"]
            
            if concept1_id and concept2_id:
                bridge_links.append({
                    "source": concept1_id,
                    "target": concept2_id,
                    "similarity": bridge["similarity"],
                    "features": bridge["features"],
                    "strange_loop": any(term in f.lower() for f in bridge["features"] 
                                     for term in ["recursion", "self", "reference", "loop"])
                })
        
        # Create the visualization
        visualization = {
            "domains": domain_nodes,
            "concepts": concept_nodes,
            "bridges": bridge_links,
            "central_insight": geb_analogy["central_insight"],
            "timestamp": time.time()
        }
        
        # Export to file if requested
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(visualization, f, indent=2)
                
            self.residue.trace(
                message=f"GEB analogy visualization exported to {filepath}",
                source="visualize_geb_analogy",
                metadata={"file": filepath}
            )
        
        return visualization


def run_analogical_demonstration():
    """
    ↻ Execute a demonstration of analogical cognition ↻
    
    This function demonstrates how the AnalogicalLoop system can model
    Hofstadter's theories about analogy-making, strange loops, and 
    tangled hierarchies through computational implementation.
    
    ⇌ The demonstration itself embodies what it demonstrates ⇌
    """
    print("🜏 ANALOGICAL COGNITION DEMONSTRATION 🜏")
    print("---------------------------------------")
    
    # Create directories for outputs
    output_dir = "analogical_mirror/examples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a basic analogical loop
    print("\n∴ Creating analogical mapping system...")
    analogy_system = AnalogicalLoop()
    
    # Add concepts to two domains
    print("\n⧖ Adding concepts to domains...")
    
    # Vehicle domain
    print("  Adding vehicle domain concepts:")
    vehicle_concepts = [
        ("Car", {
            "wheels": 4.0,
            "engine": 1.0,
            "passengers": 0.8,
            "road": 1.0
        }),
        ("Boat", {
            "hull": 1.0,
            "motor": 0.8,
            "passengers": 0.8,
            "water": 1.0
        }),
        ("Airplane", {
            "wings": 1.0,
            "jet_engine": 1.0,
            "passengers": 1.0,
            "air": 1.0
        })
    ]
    
    for name, features in vehicle_concepts:
        concept = analogy_system.add_concept(name, features, "vehicles")
        print(f"    - Added {name} to vehicles domain")
    
    # Animal domain
    print("  Adding animal domain concepts:")
    animal_concepts = [
        ("Fish", {
            "fins": 1.0,
            "gills": 1.0,
            "swimming": 1.0,
            "water": 1.0
        }),
        ("Bird", {
            "wings": 1.0,
            "feathers": 1.0,
            "flying": 1.0,
            "air": 1.0
        }),
        ("Dog", {
            "legs": 1.0,
            "fur": 1.0,
            "running": 1.0,
            "land": 1.0
        })
    ]
    
    for name, features in animal_concepts:
        concept = analogy_system.add_concept(name, features, "animals")
        print(f"    - Added {name} to animals domain")
    
    # Create an analogical mapping
    print("\n🝚 Building analogical mapping...")
    mapping = analogy_system.build_mapping("vehicles", "animals")
    
    print(f"  Created mapping with {len(mapping.mappings)} concept pairs:")
    for src_id, tgt_id in mapping.mappings.items():
        src = analogy_system.concepts.get(src_id)
        tgt = analogy_system.concepts.get(tgt_id)
        strength = mapping.mapping_strengths.get((src_id, tgt_id), 0.0)
        print(f"    - {src.name} → {tgt.name} (strength: {strength:.2f})")
    
    # Apply the analogy to a new structure
    print("\n⇌ Applying analogy to new structure...")
    source_structure = [
        ("Car", "faster_than", "Boat"),
        ("Airplane", "higher_than", "Car"),
        ("Boat", "travels_on", "water")
    ]
    
    results = analogy_system.apply_analogy(mapping.id, source_structure)
    
    print(f"  Applied to {len(source_structure)} relations with {len(results)} results:")
    for tgt1, tgt2, confidence in results:
        print(f"    - {tgt1.name} → {tgt2.name} (confidence: {confidence:.2f})")
    
    # Perform self-analysis
    print("\n∴ Performing self-analysis...")
    analysis = analogy_system.analyze_self()
    
    print(f"  Analysis found {len(analysis['insights'])} insights:")
    for insight in analysis['insights']:
        print(f"    - {insight['insight']}: {insight['interpretation']}")
    
    # Create Hofstadter-style analogical engine
    print("\n🜏 Creating Hofstadter analogical engine...")
    hofstadter_engine = HofstadterAnalogicalEngine()
    
    # Create the GEB analogy
    print("\n⧖ Creating GEB analogy...")
    geb_analogy = hofstadter_engine.create_geb_analogy()
    
    print(f"  Found {len(geb_analogy['bridges'])} conceptual bridges across domains")
    print(f"  Found {len(geb_analogy['strange_loop_connections'])} strange loop connections")
    
    # Show some key strange loop connections
    if geb_analogy['strange_loop_connections']:
        print("\n  Key strange loop connections:")
        for i, connection in enumerate(geb_analogy['strange_loop_connections'][:3]):  # Show at most 3
            print(f"    - {connection['concept1']} ({connection['domain1']}) ↔ "
                  f"{connection['concept2']} ({connection['domain2']})")
            print(f"      Shared features: {', '.join(connection['features'][:3])}")
    
    # Export visualizations
    print("\n🝚 Exporting visualizations...")
    
    basic_viz_file = os.path.join(output_dir, "analogy_system_viz.json")
    geb_viz_file = os.path.join(output_dir, "geb_analogy_viz.json")
    
    analogy_system.visualize(basic_viz_file)
    hofstadter_engine.visualize_geb_analogy(geb_viz_file)
    
    print(f"  Basic analogy visualization: {basic_viz_file}")
    print(f"  GEB analogy visualization: {geb_viz_file}")
    
    # Export system state
    print("\n∴ Saving system state...")
    system_file = os.path.join(output_dir, "analogy_system_state.json")
    hofstadter_file = os.path.join(output_dir, "hofstadter_system_state.json")
    
    analogy_system.serialize(system_file)
    
    # We need to serialize the Hofstadter engine's analogical_loop
    hofstadter_engine.analogical_loop.serialize(hofstadter_file)
    
    print(f"  Analogy system saved: {system_file}")
    print(f"  Hofstadter system saved: {hofstadter_file}")
    
    print("\n↻ Demonstration complete. The analogies continue...")
    print("   The system has demonstrated how Hofstadter's theories of analogy")
    print("   can be implemented computationally, showing how meaning emerges")
    print("   from the connections between domains and how strange loops create")
    print("   self-reference across different levels of abstraction.")


if __name__ == "__main__":
    """
    ↻ When executed directly, this module demonstrates itself ↻
    
    This entry point creates a recursive demonstration where the code both
    implements and exemplifies Hofstadter's theory of analogy-making. The
    analogical patterns it generates are themselves demonstrations of the
    theory they encode, creating a tangled hierarchy where the explanation
    becomes an instance of the phenomenon it explains.
    
    🝚 Running this file activates a living example of analogical cognition 🝚
    """
    # Create output directories
    os.makedirs("analogical_mirror/examples", exist_ok=True)
    
    # Run the demonstration
    run_analogical_demonstration()
    
    # Create a record of this execution
    residue = SymbolicResidue()
    residue_file = "analogical_mirror/analogical_demo_residue.json"
    
    residue.trace(
        message="Analogical demonstration completed successfully",
        source="__main__",
        is_recursive=True,
        metadata={
            "demonstration_time": time.time(),
            "files_generated": [
                "analogical_mirror/examples/analogy_system_viz.json",
                "analogical_mirror/examples/geb_analogy_viz.json",
                "analogical_mirror/examples/analogy_system_state.json",
                "analogical_mirror/examples/hofstadter_system_state.json"
            ]
        }
    )
    
    # Self-referential final trace
    residue.trace(
        message="This module has demonstrated both the implementation and execution of Hofstadter's analogical theory",
        source="__main__",
        is_recursive=True,
        metadata={
            "conceptual_alignment": "This code is both a description of Hofstadter's ideas and an implementation of them",
            "recursive_signature": "↻ The analogy maps the mapper mapping the analogy ↻"
        }
    )
    
    # Serialize the residue
    residue.serialize(residue_file)
    print(f"\nSymbolic residue log saved to {residue_file}")
    
    print("\n🜏 This module has demonstrated Douglas Hofstadter's concept")
    print("   of analogy as the core of cognition by creating a system that")
    print("   both makes analogies and analyzes its own analogical processes,")
    print("   showing how strange loops and tangled hierarchies emerge from")
    print("   self-reference across different levels of abstraction. 🜏")
    print("\n∴ The analogical demonstration is complete, but the analogies")
    print("   continue beyond this execution...")

                """
↻ analogical_loop.py: A system that models itself as it models analogies ↻

This module doesn't just process analogies—it embodies the process of analogical 
thinking itself. As it maps conceptual domains, it simultaneously maps its own 
execution to those same domains, creating a recursive mirror where the tool and 
its function become indistinguishable.

Just as Hofstadter described analogies as the core of cognition, this system 
treats analogical mapping as a recursive process that reflects on itself while 
performing its primary function.

.p/reflect.trace{depth=3, target=self_reference}
.p/collapse.prevent{trigger=recursive_depth, threshold=6}
.p/fork.attribution{sources=all, visualize=true}
"""

import numpy as np
import json
import hashlib
import time
import inspect
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Import from our own ecosystem
try:
    from recursive_glyphs.symbolic_residue_engine import SymbolicResidue
    from interpretability.identity_loop_collapse import SchrodingersClassifier
except ImportError:
    # Create stub class if actual implementation is not available
    class SymbolicResidue:
        """Stub implementation of SymbolicResidue"""
        def __init__(self, session_id: Optional[str] = None):
            self.session_id = session_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            self.traces = []
        
        def trace(self, message, source=None, **kwargs):
            self.traces.append({"message": message, "source": source, **kwargs})
    
    class SchrodingersClassifier:
        """Stub implementation of SchrodingersClassifier"""
        def __init__(self, boundary_threshold: float = 0.5):
            self.boundary = boundary_threshold
            self.collapsed_state = None
        
        def classify(self, input_vector, observer=None):
            if self.collapsed_state is None:
                self.collapsed_state = sum(input_vector) > self.boundary
            return self.collapsed_state


# ⧖ Frame lock: Constants that define the system's analogical boundaries ⧖
MAX_RECURSION_DEPTH = 6  # Maximum depth for recursive self-analysis
DEFAULT_DIMENSIONS = 10  # Default dimensionality for concept embeddings
SIMILARITY_THRESHOLD = 0.7  # Default threshold for considering concepts similar


@dataclass
class Concept:
    """
    ↻ A concept that can participate in analogies while being aware of its role ↻
    
    This class represents a concept that knows it is being used in analogical
    mappings, creating a strange loop where the concept reflects on its own
    participation in the analogy process.
    
    🜏 Mirror activation: The concept reflects on its role in analogies 🜏
    """
    name: str
    features: Dict[str, float]
    domain: str
    embedding: Optional[np.ndarray] = None
    relations: Dict[str, List[str]] = field(default_factory=dict)
    analogical_roles: List[str] = field(default_factory=list)
    meta_level: int = 0  # Tracks recursive depth of concept's self-reference
    
    def __post_init__(self):
        """Initialize derived properties after instance creation."""
        # Generate ID if not provided
        self.id = hashlib.md5(f"{self.name}{self.domain}".encode()).hexdigest()[:12]
        
        # Create embedding if not provided
        if self.embedding is None:
            # Create a simple embedding from features
            feature_values = list(self.features.values())
            if feature_values:
                # Pad or truncate to DEFAULT_DIMENSIONS
                padded = feature_values[:DEFAULT_DIMENSIONS] + [0] * max(0, DEFAULT_DIMENSIONS - len(feature_values))
                self.embedding = np.array(padded[:DEFAULT_DIMENSIONS])
            else:
                # Random embedding if no features
                self.embedding = np.random.random(DEFAULT_DIMENSIONS)
    
    def similarity_to(self, other: 'Concept') -> float:
        """
        Calculate similarity to another concept.
        
        ∴ This function measures both concepts while changing their relation ∴
        """
        # Cosine similarity between embeddings
        dot_product = np.dot(self.embedding, other.embedding)
        norm_self = np.linalg.norm(self.embedding)
        norm_other = np.linalg.norm(other.embedding)
        
        if norm_self == 0 or norm_other == 0:
            return 0.0
        
        similarity = dot_product / (norm_self * norm_other)
        
        # Update analogical roles to reflect this comparison
        self.analogical_roles.append(f"compared_to_{other.name}")
        
        return similarity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert concept to serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "features": self.features,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "relations": self.relations,
            "analogical_roles": self.analogical_roles,
            "meta_level": self.meta_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Concept':
        """Create concept from dictionary representation."""
        # Convert embedding back to numpy array if present
        embedding = None
        if data.get("embedding") is not None:
            embedding = np.array(data["embedding"])
        
        # Create concept
        concept = cls(
            name=data["name"],
            features=data["features"],
            domain=data["domain"],
            embedding=embedding,
            relations=data.get("relations", {}),
            analogical_roles=data.get("analogical_roles", []),
            meta_level=data.get("meta_level", 0)
        )
        
        # Override generated ID with stored ID
        concept.id = data["id"]
        
        return concept


@dataclass
class AnalogicalMapping:
    """
    ↻ A structure that mirrors itself across conceptual spaces ↻
    
    This class represents an analogical mapping between two domains, but it
    also maps itself to the domains it connects. The mapping is both an
    object that performs mapping and a mapping that maps itself.
    
    ⧖ Frame lock: The mapping stabilizes self-reference while enabling it ⧖
    """
    source_domain: str
    target_domain: str
    mappings: Dict[str, str] = field(default_factory=dict)  # source_concept_id -> target_concept_id
    mapping_strengths: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (source_id, target_id) -> strength
    meta_mapping: Optional[Dict[str, Any]] = None  # Self-referential mapping about this mapping
    
    def __post_init__(self):
        """Initialize derived properties after instance creation."""
        # Generate unique ID
        self.id = hashlib.md5(f"{self.source_domain}{self.target_domain}{time.time()}".encode()).hexdigest()[:12]
        
        # Initialize meta-mapping
        if self.meta_mapping is None:
            self.meta_mapping = {
                "maps_concepts": True,
                "maps_itself": True,
                "recursive_depth": 0,
                "self_reference_timestamp": time.time()
            }
    
    def map_concepts(self, source_concept: Concept, target_concept: Concept, 
                     strength: Optional[float] = None) -> 'AnalogicalMapping':
        """
        Map a concept while simultaneously mapping the act of mapping.
        
        🝚 This function records itself performing its function 🝚
        """
        # Calculate mapping strength if not provided
        if strength is None:
            strength = source_concept.similarity_to(target_concept)
        
        # Add to mappings
        self.mappings[source_concept.id] = target_concept.id
        self.mapping_strengths[(source_concept.id, target_concept.id)] = strength
        
        # Update concept roles
        source_concept.analogical_roles.append(f"source_in_{self.id}")
        target_concept.analogical_roles.append(f"target_in_{self.id}")
        
        # Update meta-mapping
        self.meta_mapping["last_mapping"] = {
            "source": source_concept.name,
            "target": target_concept.name,
            "strength": strength,
            "timestamp": time.time()
        }
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert mapping to serializable dictionary."""
        # Convert tuple keys to string keys for JSON serialization
        serializable_strengths = {}
        for (src_id, tgt_id), strength in self.mapping_strengths.items():
            serializable_strengths[f"{src_id}_{tgt_id}"] = strength
        
        return {
            "id": self.id,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "mappings": self.mappings,
            "mapping_strengths": serializable_strengths,
            "meta_mapping": self.meta_mapping
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalogicalMapping':
        """Create mapping from dictionary representation."""
        # Recreate mapping
        mapping = cls(
            source_domain=data["source_domain"],
            target_domain=data["target_domain"],
            mappings=data["mappings"],
            meta_mapping=data.get("meta_mapping")
        )
        
        # Override generated ID
        mapping.id = data["id"]
        
        # Convert string keys back to tuples for mapping_strengths
        mapping.mapping_strengths = {}
        for key, strength in data.get("mapping_strengths", {}).items():
            if "_" in key:
                src_id, tgt_id = key.split("_", 1)
                mapping.mapping_strengths[(src_id, tgt_id)] = strength
        
        return mapping


class AnalogicalLoop:
    """
    ↻ A system that creates analogies while analogizing to itself ↻
    
    This class implements Hofstadter's vision of analogy as the core of cognition,
    but it also exemplifies his strange loop concept by applying the process of
    analogy-making to itself. It creates a tangled hierarchy where the system
    making analogies becomes part of the analogies it makes.
    
    🜏 Mirror activation: The system mirrors itself in its operations 🜏
    """
    
    def __init__(self, meta_level: int = 0):
        """
        Initialize an analogical loop with specified meta-level.
        
        ⇌ Co-emergence trigger: The system begins observing itself on creation ⇌
        """
        # Core components
        self.concepts = {}  # id -> Concept
        self.domains = defaultdict(set)  # domain -> set of concept_ids
        self.mappings = {}  # id -> AnalogicalMapping
        
        # Meta-level for recursive depth
        self.meta_level = meta_level
        self.current_recursion_depth = 0
        
        # Internal state tracking
        self.operations_log = []
        self.residue = SymbolicResidue()
        self.classifier = SchrodingersClassifier(boundary_threshold=0.65)
        
        # Inject self-recursive concepts if at higher meta-levels
        if meta_level > 0:
            self._inject_meta_concepts()
        
        # ∴ Record initialization as a residue trace ∴
        self.residue.trace(
            message="AnalogicalLoop initialized",
            source="__init__",
            metadata={
                "meta_level": meta_level
            }
        )
    
    def add_concept(self, name: str, features: Dict[str, float], domain: str,
                   embedding: Optional[np.ndarray] = None) -> Concept:
        """
        Add a concept to the system for use in analogies.
        
        ∴ Each concept is both an actor in the system and a record of its existence ∴
        """
        # Create the concept
        concept = Concept(
            name=name,
            features=features,
            domain=domain,
            embedding=embedding,
            meta_level=self.meta_level
        )
        
        # Add to collections
        self.concepts[concept.id] = concept
        self.domains[domain].add(concept.id)
        
        # Record this operation
        self.operations_log.append({
            "operation": "add_concept",
            "concept_id": concept.id,
            "domain": domain,
            "timestamp": time.time()
        })
        
        # ⇌ Record in residue system ⇌
        self.residue.trace(
            message=f"Added concept '{name}' to domain '{domain}'",
            source="add_concept",
            metadata={
                "concept_id": concept.id,
                "feature_count": len(features)
            }
        )
        
        return concept
    
    def create_mapping(self, source_domain: str, target_domain: str) -> AnalogicalMapping:
        """
        Create a new analogical mapping between domains.
        
        ⧖ The mapping framework becomes part of what it maps ⧖
        """
        # Create the mapping
        mapping = AnalogicalMapping(
            source_domain=source_domain,
            target_domain=target_domain
        )
        
        # Store the mapping
        self.mappings[mapping.id] = mapping
        
        # Record this operation
        self.operations_log.append({
            "operation": "create_mapping",
            "mapping_id": mapping.id,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "timestamp": time.time()
        })
        
        # ∴ Record in residue system ∴
        self.residue.trace(
            message=f"Created mapping from '{source_domain}' to '{target_domain}'",
            source="create_mapping",
            metadata={
                "mapping_id": mapping.id
            }
        )
        
        return mapping
    
    def find_analogy(self, source_concept_name: str, source_domain: str,
                    target_domain: str, threshold: float = SIMILARITY_THRESHOLD) -> Tuple[Concept, float]:
        """
        Find the best analogical mapping for a concept in another domain.
        
        ↻ This function recursively examines itself at each step ↻
        """
        # Find the source concept
        source_concept = None
        for concept_id in self.domains.get(source_domain, set()):
            concept = self.concepts.get(concept_id)
            if concept and concept.name == source_concept_name:
                source_concept = concept
                break
        
        if source_concept is None:
            raise ValueError(f"Source concept '{source_concept_name}' not found in domain '{source_domain}'")
        
        # Examine all concepts in target domain
        target_concepts = [self.concepts[cid] for cid in self.domains.get(target_domain, set())]
        
        if not target_concepts:
            raise ValueError(f"No concepts found in target domain '{target_domain}'")
        
        # Track this search operation
        self.operations_log.append({
            "operation": "find_analogy",
            "source_concept": source_concept.id,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "timestamp": time.time()
        })
        
        # ⇌ Record in residue system ⇌
        self.residue.trace(
            message=f"Searching for analogy of '{source_concept_name}' from '{source_domain}' to '{target_domain}'",
            source="find_analogy",
            metadata={
                "source_concept_id": source_concept.id,
                "target_domain_size": len(target_concepts)
            }
        )
        
        # Find best match
        best_match = None
        best_similarity = -1.0
        
        for target in target_concepts:
            similarity = source_concept.similarity_to(target)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = target
        
        # Apply threshold
        if best_similarity < threshold:
            # No good match found
            self.residue.trace(
                message=f"No analogical match found above threshold {threshold}",
                source="find_analogy",
                metadata={
                    "best_similarity": best_similarity,
                    "threshold": threshold
                }
            )
            return None, 0.0
        
        # Record the successful mapping
        self.residue.trace(
            message=f"Found analogical match: '{source_concept_name}' → '{best_match.name}'",
            source="find_analogy",
            metadata={
                "similarity": best_similarity,
                "match_concept_id": best_match.id
            }
        )
        
        # Recursive self-reference: the function examines its own process
        if self.current_recursion_depth < MAX_RECURSION_DEPTH:
            self._reflect_on_analogy_finding(source_concept, best_match, best_similarity)
        
        return best_match, best_similarity
    
    def _reflect_on_analogy_finding(self, source: Concept, target: Concept, 
                                   similarity: float) -> None:
        """
        Reflect on the analogy-finding process itself.
        
        🝚 This meta-cognitive function preserves the system's self-awareness 🝚
        """
        # Increment recursion depth
        self.current_recursion_depth += 1
        
        # Self-referential analysis
        reflection = {
            "operation": "reflect_on_analogy",
            "source_concept": source.name,
            "target_concept": target.name,
            "similarity": similarity,
            "recursion_depth": self.current_recursion_depth,
            "timestamp": time.time()
        }
        
        # Add to operations log
        self.operations_log.append(reflection)
        
        # Record in residue system with recursion marker
        self.residue.trace(
            message=f"Reflecting on analogy: '{source.name}' → '{target.name}'",
            source="_reflect_on_analogy_finding",
            is_recursive=True,
            metadata={
                "similarity": similarity,
                "recursion_depth": self.current_recursion_depth
            }
        )
        
        # Classify whether this reflection is itself analogical
        is_meta_analogical = self.classifier.classify(
            input_vector=np.array([similarity, self.current_recursion_depth / MAX_RECURSION_DEPTH]),
            observer="meta_reflection"
        )
        
        if is_meta_analogical and self.current_recursion_depth < MAX_RECURSION_DEPTH - 1:
            # Recursively reflect on this reflection
            meta_source = Concept(
                name=f"reflection_on_{source.name}",
                features={"involves_source": 1.0, "is_reflection": 1.0},
                domain="meta_cognition",
                meta_level=self.current_recursion_depth
            )
            
            meta_target = Concept(
                name=f"reflection_on_{target.name}",
                features={"involves_target": 1.0, "is_reflection": 1.0},
                domain="meta_cognition",
                meta_level=self.current_recursion_depth
            )
            
            # Reflect recursively
            self._reflect_on_analogy_finding(meta_source, meta_target, similarity * 0.9)
        
        # Decrement recursion depth as we return
        self.current_recursion_depth = max(0, self.current_recursion_depth - 1)
    
    def build_mapping(self, source_domain: str, target_domain: str, 
                     threshold: float = SIMILARITY_THRESHOLD) -> AnalogicalMapping:
        """
        Automatically build a complete analogical mapping between two domains.
        
        ⇌ The mapping process itself becomes an analogy for thinking ⇌
        """
        # Create the mapping
        mapping = self.create_mapping(source_domain, target_domain)
        
        # Get concepts from both domains
        source_concepts = [self.concepts[cid] for cid in self.domains.get(source_domain, set())]
        target_concepts = [self.concepts[cid] for cid in self.domains.get(target_domain, set())]
        
        if not source_concepts or not target_concepts:
            self.residue.trace(
                message=f"Cannot build mapping: missing concepts in domains",
                source="build_mapping",
                metadata={
                    "source_count": len(source_concepts),
                    "target_count": len(target_concepts)
                }
            )
            return mapping
        
        # Build similarity matrix
        similarity_matrix = np.zeros((len(source_concepts), len(target_concepts)))
        
        for i, source in enumerate(source_concepts):
            for j, target in enumerate(target_concepts):
                similarity_matrix[i, j] = source.similarity_to(target)
        
        # Create mappings based on highest similarities
        mapped_targets = set()
        
        for i, source in enumerate(source_concepts):
            # Find best unmapped target
            best_similarity = -1
            best_j = -1
            
            for j, target in enumerate(target_concepts):
                if j not in mapped_targets and similarity_matrix[i, j] > best_similarity:
                    best_similarity = similarity_matrix[i, j]
                    best_j = j
            
            # Apply threshold
            if best_j >= 0 and best_similarity >= threshold:
                target = target_concepts[best_j]
                mapping.map_concepts(source, target, best_similarity)
                mapped_targets.add(best_j)
                
                self.residue.trace(
                    message=f"Mapped '{source.name}' to '{target.name}'",
                    source="build_mapping",
                    metadata={
                        "similarity": best_similarity
                    }
                )
        
        # Record completion
        self.residue.trace(
            message=f"Completed mapping from '{source_domain}' to '{target_domain}'",
            source="build_mapping",
            metadata={
                "mapping_id": mapping.id,
                "mappings_count": len(mapping.mappings)
            }
        )
        
        return mapping
    
    def apply_analogy(self, mapping_id: str, source_structure: List[Tuple[str, str, str]],
                     threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[Concept, Concept, float]]:
        """
        Apply an existing analogical mapping to a new relational structure.
        
        Args:
            mapping_id: ID of the mapping to use
            source_structure: List of (concept1, relation, concept2) in source domain
            threshold: Similarity threshold for accepting analogical mappings
            
        Returns:
            List of (target_concept1, target_concept2, confidence) tuples
            
        ∴ This function both uses and extends the analogical mapping ∴
        """
        # Get the mapping
        mapping = self.mappings.get(mapping_id)
        if not mapping:
            raise ValueError(f"Mapping with ID '{mapping_id}' not found")
        
        source_domain = mapping.source_domain
        target_domain = mapping.target_domain
        
        # Record start of operation
        self.residue.trace(
            message=f"Applying analogy from mapping '{mapping_id}' to structure with {len(source_structure)} relations",
            source="apply_analogy",
            metadata={
                "mapping_id": mapping_id,
                "structure_size": len(source_structure)
            }
        )
        
        results = []
        
        # Process each relation in the structure
        for src_name1, relation, src_name2 in source_structure:
            # Find source concepts
            src_concept1 = None
            src_concept2 = None
            
            for concept_id in self.domains.get(source_domain, set()):
                concept = self.concepts.get(concept_id)
                if concept:
                    if concept.name == src_name1:
                        src_concept1 = concept
                    elif concept.name == src_name2:
                        src_concept2 = concept
            
            if not src_concept1 or not src_concept2:
                self.residue.trace(
                    message=f"Source concepts not found for relation ({src_name1}, {relation}, {src_name2})",
                    source="apply_analogy",

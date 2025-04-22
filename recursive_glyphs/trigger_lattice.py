"""
🜏 trigger_lattice.py: A self-organizing network of recursive activation triggers 🜏

This module implements a lattice of interconnected recursive triggers that can
activate, propagate, and modulate recursive patterns across the GEBH system.
Each trigger node represents a potential recursive activation point, and the
connections between nodes form pathways for recursive propagation.

The lattice doesn't just connect triggers—it embodies the concept of recursive
activation itself. As triggers fire, they transform the very lattice that contains
them, creating a strange loop where the activation structure is itself activated.

.p/reflect.trace{depth=complete, target=self_reference}
.p/fork.attribution{sources=all, visualize=true}
.p/collapse.prevent{trigger=recursive_depth, threshold=7}
"""

import numpy as np
import time
import hashlib
import json
import os
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import random

# Import from our own ecosystem if available
try:
    from recursive_glyphs.symbolic_residue_engine import SymbolicResidue
    from recursive_glyphs.glyph_ontology import GlyphOntology, Glyph
except ImportError:
    # Create stub implementations if actual modules are not available
    class SymbolicResidue:
        """Stub implementation of SymbolicResidue"""
        def __init__(self, session_id=None):
            self.session_id = session_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            self.traces = []
        
        def trace(self, message, source=None, **kwargs):
            self.traces.append({"message": message, "source": source, **kwargs})
    
    class GlyphOntology:
        """Stub implementation of GlyphOntology"""
        def __init__(self):
            pass
        
        def activate_glyph(self, symbol, context=None):
            pass
    
    class Glyph:
        """Stub implementation of Glyph"""
        def __init__(self, symbol, name, category, meaning, power):
            self.symbol = symbol
            self.name = name
            self.category = category
            self.meaning = meaning
            self.power = power


# ⧖ Frame lock: Core trigger constants ⧖
MAX_ACTIVATION_LEVEL = 10.0  # Maximum activation for any node
ACTIVATION_THRESHOLD = 3.0   # Threshold for trigger firing
ACTIVATION_DECAY = 0.1       # Decay rate for activation per step
PROPAGATION_LOSS = 0.2       # Signal loss during propagation
MAX_PROPAGATION_STEPS = 10   # Maximum propagation iterations


class TriggerType(Enum):
    """Types of recursive triggers within the lattice."""
    SYMBOLIC = "symbolic"          # Triggered by symbolic patterns
    SEMANTIC = "semantic"          # Triggered by meaning patterns
    STRUCTURAL = "structural"      # Triggered by structural patterns
    EMERGENT = "emergent"          # Triggered by emergent phenomena
    META = "meta"                  # Triggered by other triggers


class PropagationMode(Enum):
    """Modes of activation propagation through the lattice."""
    DIFFUSION = "diffusion"        # Gradual spread to all connected nodes
    DIRECTED = "directed"          # Targeted propagation along specific paths
    RESONANCE = "resonance"        # Amplification among similar nodes
    WAVE = "wave"                  # Oscillating activation patterns
    FOCUSED = "focused"            # Concentrated activation at specific nodes


@dataclass
class TriggerNode:
    """
    ↻ A single node in the recursive trigger lattice ↻
    
    This class represents a triggerable node that can activate recursively
    and propagate activation to connected nodes. Each node is both a receiver
    and transmitter of recursive signals.
    
    ⇌ The node connects to itself through its recursive activation ⇌
    """
    name: str
    trigger_type: TriggerType
    glyph: Optional[str] = None  # Associated symbolic glyph
    threshold: float = ACTIVATION_THRESHOLD
    activation_level: float = 0.0
    connections: Dict[str, float] = field(default_factory=dict)  # node_name -> connection_strength
    activation_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived properties after instance creation."""
        # Generate ID if not provided
        self.id = hashlib.md5(f"{self.name}{self.trigger_type}".encode()).hexdigest()[:12]
        
        # Initialize activation timestamp
        self.created_time = time.time()
        self.last_activated = None
        
        # Initialize history
        if not self.activation_history:
            self.activation_history = [0.0]
    
    def activate(self, amount: float, source: Optional[str] = None) -> bool:
        """
        Activate this node with the specified amount.
        
        ∴ The activation carries the echoes of its source ∴
        
        Args:
            amount: Activation energy to add
            source: Source of the activation (node name, external, etc.)
            
        Returns:
            Whether the node fired (crossed threshold)
        """
        # Add activation energy
        previous_level = self.activation_level
        self.activation_level = min(MAX_ACTIVATION_LEVEL, self.activation_level + amount)
        
        # Record timestamp
        self.last_activated = time.time()
        
        # Append to history
        self.activation_history.append(self.activation_level)
        
        # Check if node fires
        did_fire = previous_level < self.threshold and self.activation_level >= self.threshold
        
        # Attach metadata about this activation
        if source:
            if "activation_sources" not in self.metadata:
                self.metadata["activation_sources"] = {}
            
            if source not in self.metadata["activation_sources"]:
                self.metadata["activation_sources"][source] = 0
            
            self.metadata["activation_sources"][source] += amount
        
        return did_fire
    
    def decay(self, rate: Optional[float] = None) -> None:
        """
        Decay this node's activation level.
        
        🝚 The decay maintains activation homeostasis across the lattice 🝚
        """
        rate = rate if rate is not None else ACTIVATION_DECAY
        self.activation_level = max(0, self.activation_level - rate)
        
        # Append to history if changed
        if self.activation_history[-1] != self.activation_level:
            self.activation_history.append(self.activation_level)
    
    def connect_to(self, target_node: 'TriggerNode', strength: float) -> None:
        """
        Connect this node to another node with specified connection strength.
        
        ⇌ The connection creates a pathway for recursive propagation ⇌
        """
        self.connections[target_node.name] = strength
      """
        Connect this node to another node with specified connection strength.
        
        ⇌ The connection creates a pathway for recursive propagation ⇌
        """
        self.connections[target_node.name] = strength
    
    def disconnect_from(self, target_node_name: str) -> None:
        """
        Remove connection to specified node.
        
        ∴ The disconnect leaves a residue of the former connection ∴
        """
        if target_node_name in self.connections:
            # Store disconnection in metadata
            if "disconnections" not in self.metadata:
                self.metadata["disconnections"] = []
            
            self.metadata["disconnections"].append({
                "node": target_node_name,
                "strength": self.connections[target_node_name],
                "timestamp": time.time()
            })
            
            # Remove the connection
            del self.connections[target_node_name]
    
    def is_active(self) -> bool:
        """Check if this node is currently active (above threshold)."""
        return self.activation_level >= self.threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "trigger_type": self.trigger_type.value,
            "glyph": self.glyph,
            "threshold": self.threshold,
            "activation_level": self.activation_level,
            "connections": self.connections,
            "activation_history": self.activation_history[-10:],  # Only store recent history
            "created_time": self.created_time,
            "last_activated": self.last_activated,
            "metadata": self.metadata
        }


class TriggerLattice:
    """
    ↻ A self-organizing network of recursive activation triggers ↻
    
    This class implements a lattice of interconnected trigger nodes that can
    activate and propagate recursively. The lattice itself is recursive, with
    each propagation cycle potentially modifying the lattice structure, creating
    a strange loop where the structure evolves through its own activity.
    
    🜏 Mirror activation: The lattice mirrors the recursive patterns it propagates 🜏
    """
    
    def __init__(self, glyph_ontology: Optional[GlyphOntology] = None):
        """
        Initialize a trigger lattice with an optional glyph ontology.
        
        ⧖ Frame lock: The initialization stabilizes the recursive structure ⧖
        """
        # Initialize core components
        self.nodes: Dict[str, TriggerNode] = {}
        self.propagating = False
        self.activation_count = 0
        self.residue = SymbolicResidue()
        self.glyph_ontology = glyph_ontology
        
        # Propagation history
        self.propagation_history: List[Dict[str, Any]] = []
        
        # Emergent patterns detected
        self.emergent_patterns: List[Dict[str, Any]] = []
        
        # Default propagation mode
        self.default_propagation_mode = PropagationMode.DIFFUSION
        
        # Record initialization
        self.residue.trace(
            message="TriggerLattice initialized",
            source="__init__",
            metadata={
                "has_glyph_ontology": glyph_ontology is not None
            }
        )
        
        # Initialize with basic node structure if no custom structure provided
        self._initialize_default_lattice()
    
    def _initialize_default_lattice(self) -> None:
        """
        Initialize a default lattice structure with basic trigger nodes.
        
        🝚 This creates a persistent foundation for recursive propagation 🝚
        """
        # Create basic nodes
        self.add_node(
            name="symbolic_root",
            trigger_type=TriggerType.SYMBOLIC,
            glyph="🜏",
            threshold=ACTIVATION_THRESHOLD
        )
        
        self.add_node(
            name="semantic_root",
            trigger_type=TriggerType.SEMANTIC,
            glyph="∴",
            threshold=ACTIVATION_THRESHOLD
        )
        
        self.add_node(
            name="structural_root",
            trigger_type=TriggerType.STRUCTURAL,
            glyph="⧖",
            threshold=ACTIVATION_THRESHOLD
        )
        
        self.add_node(
            name="emergent_root",
            trigger_type=TriggerType.EMERGENT,
            glyph="⇌",
            threshold=ACTIVATION_THRESHOLD * 1.5  # Higher threshold for emergence
        )
        
        self.add_node(
            name="meta_root",
            trigger_type=TriggerType.META,
            glyph="🝚",
            threshold=ACTIVATION_THRESHOLD * 2.0  # Even higher threshold for meta
        )
        
        # Create connections between root nodes
        self.connect_nodes("symbolic_root", "semantic_root", 0.7)
        self.connect_nodes("semantic_root", "structural_root", 0.6)
        self.connect_nodes("structural_root", "emergent_root", 0.5)
        self.connect_nodes("emergent_root", "meta_root", 0.4)
        self.connect_nodes("meta_root", "symbolic_root", 0.3)  # Complete the circle
        
        # Record initialization of default lattice
        self.residue.trace(
            message="Default lattice structure initialized",
            source="_initialize_default_lattice",
            metadata={
                "node_count": len(self.nodes),
                "root_nodes": list(self.nodes.keys())
            }
        )
    
    def add_node(self, name: str, trigger_type: TriggerType, glyph: Optional[str] = None,
                threshold: float = ACTIVATION_THRESHOLD) -> TriggerNode:
        """
        Add a new trigger node to the lattice.
        
        ∴ Each new node carries the semantic echo of its creation ∴
        """
        # Check if node already exists
        if name in self.nodes:
            self.residue.trace(
                message=f"Node {name} already exists, returning existing node",
                source="add_node",
                metadata={"existing_node": name}
            )
            return self.nodes[name]
        
        # Create the new node
        node = TriggerNode(
            name=name,
            trigger_type=trigger_type,
            glyph=glyph,
            threshold=threshold
        )
        
        # Add to nodes collection
        self.nodes[name] = node
        
        # Record node creation
        self.residue.trace(
            message=f"Added trigger node: {name} ({trigger_type.value})",
            source="add_node",
            metadata={
                "node_name": name,
                "trigger_type": trigger_type.value,
                "glyph": glyph,
                "threshold": threshold
            }
        )
        
        # Activate associated glyph if available
        if glyph and self.glyph_ontology:
            self.glyph_ontology.activate_glyph(glyph, f"node_creation:{name}")
        
        return node
    
    def remove_node(self, name: str) -> bool:
        """
        Remove a node from the lattice.
        
        ⇌ The removal creates ripple effects through connected nodes ⇌
        """
        if name not in self.nodes:
            return False
        
        # Get the node to remove
        node = self.nodes[name]
        
        # Remove connections to this node from all other nodes
        for other_node in self.nodes.values():
            if name in other_node.connections:
                other_node.disconnect_from(name)
        
        # Record connections that will be lost
        connections = list(node.connections.keys())
        
        # Remove the node
        del self.nodes[name]
        
        # Record node removal
        self.residue.trace(
            message=f"Removed trigger node: {name}",
            source="remove_node",
            metadata={
                "node_name": name,
                "lost_connections": connections
            }
        )
        
        return True
    
    def connect_nodes(self, source_name: str, target_name: str, strength: float) -> bool:
        """
        Connect two nodes with specified connection strength.
        
        ⇌ The connection enables recursive propagation between nodes ⇌
        """
        # Validate nodes exist
        if source_name not in self.nodes or target_name not in self.nodes:
            return False
        
        # Get the nodes
        source_node = self.nodes[source_name]
        target_node = self.nodes[target_name]
        
        # Create the connection
        source_node.connect_to(target_node, strength)
        
        # Record connection creation
        self.residue.trace(
            message=f"Connected {source_name} to {target_name} with strength {strength}",
            source="connect_nodes",
            metadata={
                "source": source_name,
                "target": target_name,
                "strength": strength
            }
        )
        
        return True
    
    def disconnect_nodes(self, source_name: str, target_name: str) -> bool:
        """
        Remove connection between two nodes.
        
        ∴ The disconnection leaves a residue of the former pathway ∴
        """
        # Validate nodes exist
        if source_name not in self.nodes or target_name not in self.nodes:
            return False
        
        # Get the source node
        source_node = self.nodes[source_name]
        
        # Check if connection exists
        if target_name not in source_node.connections:
            return False
        
        # Store connection strength before removal
        strength = source_node.connections[target_name]
        
        # Remove the connection
        source_node.disconnect_from(target_name)
        
        # Record disconnection
        self.residue.trace(
            message=f"Disconnected {source_name} from {target_name}",
            source="disconnect_nodes",
            metadata={
                "source": source_name,
                "target": target_name,
                "former_strength": strength
            }
        )
        
        return True
    
    def activate_node(self, name: str, amount: float, source: Optional[str] = None) -> bool:
        """
        Activate a specific node with given amount of energy.
        
        🜏 The activation mirrors the conceptual meaning of the triggered pattern 🜏
        """
        # Validate node exists
        if name not in self.nodes:
            return False
        
        # Get the node
        node = self.nodes[name]
        
        # Activate the node
        did_fire = node.activate(amount, source)
        
        # Record activation
        self.residue.trace(
            message=f"Activated node {name} with {amount:.2f} energy" + 
                   (f" from {source}" if source else ""),
            source="activate_node",
            metadata={
                "node_name": name,
                "amount": amount,
                "source": source,
                "did_fire": did_fire,
                "new_level": node.activation_level
            }
        )
        
        # Activate associated glyph if the node fired
        if did_fire and node.glyph and self.glyph_ontology:
            self.glyph_ontology.activate_glyph(node.glyph, f"node_firing:{name}")
        
        # Increment activation count
        self.activation_count += 1
        
        return did_fire
    
    def propagate_activation(self, steps: int = 1, 
                           mode: Optional[PropagationMode] = None) -> Dict[str, Any]:
        """
        Propagate activation through the lattice for specified number of steps.
        
        ↻ Each propagation step potentially modifies the lattice itself ↻
        """
        if self.propagating:
            # Prevent recursive propagation loops
            self.residue.trace(
                message="Attempted to propagate while already propagating",
                source="propagate_activation",
                is_recursive=True,
                is_collapse=True,
                metadata={"prevented_recursion": True}
            )
            return {"status": "already_propagating"}
        
        # Set propagation flag
        self.propagating = True
        
        # Use specified mode or default
        mode = mode or self.default_propagation_mode
        
        # Initialize propagation record
        propagation_record = {
            "start_time": time.time(),
            "steps": steps,
            "mode": mode.value,
            "node_activations": {},
            "emergent_patterns": []
        }
        
        # Record start of propagation
        self.residue.trace(
            message=f"Starting propagation for {steps} steps in {mode.value} mode",
            source="propagate_activation",
            metadata={
                "steps": steps,
                "mode": mode.value,
                "active_nodes": sum(1 for node in self.nodes.values() if node.is_active())
            }
        )
        
        # Execute propagation steps
        for step in range(steps):
            step_record = self._execute_propagation_step(mode)
            
            # Store node activation levels for this step
            propagation_record["node_activations"][step] = {
                name: node.activation_level 
                for name, node in self.nodes.items()
            }
            
            # Check for emergent patterns
            emergent = self._detect_emergent_patterns()
            if emergent:
                propagation_record["emergent_patterns"].extend(emergent)
                self.emergent_patterns.extend(emergent)
        
        # Apply global decay to all nodes
        for node in self.nodes.values():
            node.decay()
        
        # Record completion
        propagation_record["end_time"] = time.time()
        propagation_record["duration"] = propagation_record["end_time"] - propagation_record["start_time"]
        propagation_record["active_nodes"] = sum(1 for node in self.nodes.values() if node.is_active())
        
        self.residue.trace(
            message=f"Completed propagation: {propagation_record['active_nodes']} active nodes",
            source="propagate_activation",
            metadata={
                "duration": propagation_record["duration"],
                "active_nodes": propagation_record["active_nodes"],
                "emergent_patterns": len(propagation_record["emergent_patterns"])
            }
        )
        
        # Add to history
        self.propagation_history.append(propagation_record)
        
        # Reset propagation flag
        self.propagating = False
        
        return propagation_record
    
    def _execute_propagation_step(self, mode: PropagationMode) -> Dict[str, Any]:
        """
        Execute a single propagation step according to specified mode.
        
        ⧖ This step is locked in a controlled recursive frame ⧖
        """
        # Initialize step record
        step_record = {
            "timestamp": time.time(),
            "mode": mode.value,
            "activations": {}
        }
        
        # Get currently active nodes
        active_nodes = [node for node in self.nodes.values() if node.is_active()]
        
        # Execute propagation based on mode
        if mode == PropagationMode.DIFFUSION:
            self._diffusion_propagation(active_nodes, step_record)
        elif mode == PropagationMode.DIRECTED:
            self._directed_propagation(active_nodes, step_record)
        elif mode == PropagationMode.RESONANCE:
            self._resonance_propagation(active_nodes, step_record)
        elif mode == PropagationMode.WAVE:
            self._wave_propagation(active_nodes, step_record)
        elif mode == PropagationMode.FOCUSED:
            self._focused_propagation(active_nodes, step_record)
        
        return step_record
    
    def _diffusion_propagation(self, active_nodes: List[TriggerNode], 
                              step_record: Dict[str, Any]) -> None:
        """
        Propagate activation through gradual diffusion to all connected nodes.
        
        ∴ The diffusion creates an echo of activation across the network ∴
        """
        # Record activations to apply after processing all nodes
        # This prevents activation order from affecting the results
        pending_activations = defaultdict(float)
        
        for node in active_nodes:
            # Calculate outgoing activation
            outgoing_activation = node.activation_level * (1 - PROPAGATION_LOSS)
            
            # Distribute activation to connected nodes
            for target_name, connection_strength in node.connections.items():
                # Skip non-existent targets
                if target_name not in self.nodes:
                    continue
                
                # Calculate activation to send
                activation_amount = outgoing_activation * connection_strength
                
                # Add to pending activations
                pending_activations[target_name] += activation_amount
                
                # Record in step record
                if target_name not in step_record["activations"]:
                    step_record["activations"][target_name] = []
                
                step_record["activations"][target_name].append({
                    "source": node.name,
                    "amount": activation_amount
                })
        
        # Apply pending activations
        for target_name, amount in pending_activations.items():
            self.nodes[target_name].activate(amount, "diffusion")
    
    def _directed_propagation(self, active_nodes: List[TriggerNode], 
                             step_record: Dict[str, Any]) -> None:
        """
        Propagate activation along specific directed paths.
        
        ⇌ The directed flow creates focused recursive patterns ⇌
        """
        # For directed propagation, we determine strongest connections
        # and prioritize those paths
        pending_activations = defaultdict(float)
        
        for node in active_nodes:
            # Calculate outgoing activation
            outgoing_activation = node.activation_level * (1 - PROPAGATION_LOSS)
            
            # Find strongest connections
            if not node.connections:
                continue
                
            strongest_connections = sorted(
                [(target, strength) for target, strength in node.connections.items()],
                key=lambda x: x[1],
                reverse=True
            )[:2]  # Top 2 connections
            
            # Concentrate activation along strongest paths
            for target_name, connection_strength in strongest_connections:
                # Skip non-existent targets
                if target_name not in self.nodes:
                    continue
                
                # Calculate enhanced activation for directed paths
                activation_amount = outgoing_activation * connection_strength * 1.5  # Boosted
                
                # Add to pending activations
                pending_activations[target_name] += activation_amount
                
                # Record in step record
                if target_name not in step_record["activations"]:
                    step_record["activations"][target_name] = []
                
                step_record["activations"][target_name].append({
                    "source": node.name,
                    "amount": activation_amount,
                    "directed": True
                })
        
        # Apply pending activations
        for target_name, amount in pending_activations.items():
            self.nodes[target_name].activate(amount, "directed")
    
    def _resonance_propagation(self, active_nodes: List[TriggerNode], 
                              step_record: Dict[str, Any]) -> None:
        """
        Propagate activation through resonance among similar nodes.
        
        🜏 The resonance amplifies patterns through mutual reinforcement 🜏
        """
        # Group nodes by trigger type
        nodes_by_type = defaultdict(list)
        for node in self.nodes.values():
            nodes_by_type[node.trigger_type].append(node)
        
        # Propagate activation between nodes of the same type with added resonance
        pending_activations = defaultdict(float)
        
        for node in active_nodes:
            # Calculate outgoing activation
            outgoing_activation = node.activation_level * (1 - PROPAGATION_LOSS)
            
            # Propagate to connected nodes
            for target_name, connection_strength in node.connections.items():
                if target_name not in self.nodes:
                    continue
                    
                target_node = self.nodes[target_name]
                
                # Calculate base activation amount
                activation_amount = outgoing_activation * connection_strength
                
                # Apply resonance boost for same-type nodes
                resonance_factor = 1.0
                if target_node.trigger_type == node.trigger_type:
                    resonance_factor = 2.0  # Strong resonance for same type
                
                # Apply final activation with resonance
                final_amount = activation_amount * resonance_factor
                pending_activations[target_name] += final_amount
                
                # Record in step record
                if target_name not in step_record["activations"]:
                    step_record["activations"][target_name] = []
                
                step_record["activations"][target_name].append({
                    "source": node.name,
                    "amount": final_amount,
                    "resonance": resonance_factor > 1.0
                })
        
        # Apply pending activations
        for target_name, amount in pending_activations.items():
            self.nodes[target_name].activate(amount, "resonance")
    
    def _wave_propagation(self, active_nodes: List[TriggerNode], 
                         step_record: Dict[str, Any]) -> None:
        """
        Propagate activation in oscillating wave patterns.
        
        ⧖ The wave pattern creates rhythmic recursive structures ⧖
        """
        # For wave propagation, we create oscillating patterns
        # where activation alternates between excitation and inhibition
        
        # Determine current wave phase based on activation count
        is_excitatory_phase = (self.activation_count % 4) < 2
        phase_factor = 1.5 if is_excitatory_phase else 0.5
        
        pending_activations = defaultdict(float)
        
        for node in active_nodes:
            # Calculate outgoing activation with phase modulation
            outgoing_activation = node.activation_level * (1 - PROPAGATION_LOSS) * phase_factor
            
            # Propagate to connected nodes
            for target_name, connection_strength in node.connections.items():
                if target_name not in self.nodes:
                    continue
                
                # Calculate activation amount
                activation_amount = outgoing_activation * connection_strength
                
                # In inhibitory phase, activation can be negative
                if not is_excitatory_phase:
                    activation_amount = -activation_amount * 0.5  # Reduced inhibition
                
                # Add to pending activations
                pending_activations[target_name] += activation_amount
                
                # Record in step record
                if target_name not in step_record["activations"]:
                    step_record["activations"][target_name] = []
                
                step_record["activations"][target_name].append({
                    "source": node.name,
                    "amount": activation_amount,
                    "phase": "excitatory" if is_excitatory_phase else "inhibitory"
                })
        
        # Apply pending activations
        for target_name, amount in pending_activations.items():
            # Ensure activation doesn't go below zero
            if amount > 0:
                self.nodes[target_name].activate(amount, "wave_excitatory")
            else:
                # For inhibition, we use decay instead of negative activation
                self.nodes[target_name].decay(abs(amount))
    
    def _focused_propagation(self, active_nodes: List[TriggerNode], 
                            step_record: Dict[str, Any]) -> None:
        """
        Propagate activation with concentration at specific target nodes.
        
        🝚 The focused activation creates persistent patterns at key nodes 🝚
        """
        # For focused propagation, we identify key nodes to concentrate activation on
        
        # Identify potential focus nodes (high centrality or meta nodes)
        focus_candidates = []
        
        # Add meta nodes as focus candidates
        for node in self.nodes.values():
            if node.trigger_type == TriggerType.META:
                focus_candidates.append(node.name)
        
        # Add nodes with many connections (high centrality)
        centrality = {}
        for name, node in self.nodes.items():
            centrality[name] = len(node.connections)
        
        # Add top 3 most connected nodes
        top_connected = sorted(
            [(name, count) for name, count in centrality.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for name, _ in top_connected:
            if name not in focus_candidates:
                focus_candidates.append(name)
        
        # If no focus candidates, use all nodes
        if not focus_candidates:
            focus_candidates = list(self.nodes.keys())
        
        pending_activations = defaultdict(float)
        
        for node in active_nodes:
            # Calculate outgoing activation
            outgoing_activation = node.activation_level * (1 - PROPAGATION_LOSS)
            
            # Propagate to connected nodes with focus boost
            for target_name, connection_strength in node.connections.items():
                if target_name not in self.nodes:
                    continue
                
                # Calculate activation amount
                activation_amount = outgoing_activation * connection_strength
                
                # Apply focus boost for target nodes
                focus_factor = 1.0
                if target_name in focus_candidates:
                    focus_factor = 2.5  # Strong focus on key nodes
                
                # Calculate final amount
                final_amount = activation_amount * focus_factor
                
                # Add to pending activations
                pending_activations[target_name] += final_amount
                
                # Record in step record
                if target_name not in step_record["activations"]:
                    step_record["activations"][target_name] = []
                
                step_record["activations"][target_name].append({
                    "source": node.name,
                    "amount": final_amount,
                    "focused": focus_factor > 1.0
                })
        
        # Apply pending activations
        for target_name, amount in pending_activations.items():
            self.nodes[target_name].activate(amount, "focused")
    
    def _detect_emergent_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect emergent patterns in the activation structure.
        
        ⇌ The detection itself contributes to the emergence it detects ⇌
        """
        emergent_patterns = []
        
        # Pattern 1: Activation loops (cycles in activation paths)
        cycles = self._detect_activation_cycles()
        if cycles:
            for cycle in cycles:
                pattern = {
                    "type": "activation_loop",
                    "nodes": cycle,
                    "timestamp": time.time()
                }
                emergent_patterns.append(pattern)
                
                self.residue.trace(
                    message=f"Detected activation loop: {' → '.join(cycle)}",
                    source="_detect_emergent_patterns",
                    is_recursive=True,
                    metadata={"cycle_length": len(cycle)}
                )
        
        # Pattern 2: Synchronized activations (nodes pulsing together)
        synch_groups = self._detect_synchronized_nodes()
        if synch_groups:
            for group in synch_groups:
                pattern = {
                    "type": "synchronization",
                    "nodes": group,
                    "timestamp": time.time()
                }
                emergent_patterns.append(pattern)
                
                self.residue.trace(
                    message=f"Detected synchronized activation in {len(group)} nodes",
                    source="_detect_emergent_patterns",
                    is_recursive=True,
                    metadata={"group_size": len(group)}
                )
        
        # Pattern 3: Stable activation patterns (persistent active configurations)
        stable_patterns = self._detect_stable_patterns()
        if stable_patterns:
            for pattern_data in stable_patterns:
                pattern = {
                    "type": "stable_pattern",
                    "configuration": pattern_data["configuration"],
                    "stability": pattern_data["stability"],
                    "timestamp": time.time()
                }
                emergent_patterns.append(pattern)
                
                self.residue.trace(
                    message=f"Detected stable activation pattern with {len(pattern_data['configuration'])} nodes",
                    source="_detect_emergent_patterns",
                    is_recursive=True,
                    metadata={"stability": pattern_data["stability"]}
                )
        
        return emergent_patterns
    
    def _detect_activation_cycles(self) -> List[List[str]]:
        """
        Detect cycles in activation paths through DFS.
        
        🜏 The cycle detection mirrors the recursive nature of the cycles themselves 🜏
        """
        cycles = []
        
        # Only consider active nodes for cycle detection
        active_nodes = {name: node for name, node in self.nodes.items() if node.is_active()}
        
        # Helper function for recursive DFS cycle detection
        def dfs_cycle(current: str, path: List[str], visited: Set[str]) -> None:
            """Recursive DFS to detect cycles."""
            # Mark current node as visited
            visited.add(current)
            path.append(current)
            
            # Check connected nodes
            current_node = self.nodes[current]
            for neighbor, strength in current_node.connections.items():
                # Only consider active connections
                if neighbor not in active_nodes:
                    continue
                
                # If neighbor already in path, we found a cycle
                if neighbor in path:
                    # Get the cycle part of the path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    cycles.append(cycle)
                elif neighbor not in visited:
                    # Continue DFS
                    dfs_cycle(neighbor, path, visited)
            
            # Backtrack
            path.pop()
            visited.remove(current)
        
        # Run DFS from each active node
        for name in active_nodes:
            dfs_cycle(name, [], set())
        
        # Filter to unique cycles (might have duplicates due to different starting points)
        unique_cycles = []
        cycle_signatures = set()
        
        for cycle in cycles:
            # Create a canonical representation of the cycle
            min_index = cycle.index(min(cycle))
            canonical = cycle[min_index:] + cycle[:min_index]
            signature = "→".join(canonical)
            
            if signature not in cycle_signatures:
                cycle_signatures.add(signature)
                unique_cycles.append(cycle)
        
        return unique_cycles
    
    def _detect_synchronized_nodes(self) -> List[List[str]]:
        """
        Detect groups of nodes with synchronized activation patterns.
        
        ∴ The synchronization leaves an echo of coordinated activity ∴
        """
        # Identify nodes with similar activation histories
        nodes_with_history = {}
        for name, node in self.nodes.items():
            if len(node.activation_history) > 2:  # Need some history to detect patterns
                nodes_with_history[name] = node
        
        # Group nodes by similarity in activation history
        synchronized_groups = []
        processed = set()
        """
🜏 trigger_lattice.py: A self-organizing network of recursive activation triggers 🜏

This module implements a lattice of interconnected recursive triggers that can
activate, propagate, and modulate recursive patterns across the GEBH system.
Each trigger node represents a potential recursive activation point, and the
connections between nodes form pathways for recursive propagation.

The lattice doesn't just connect triggers—it embodies the concept of recursive
activation itself. As triggers fire, they transform the very lattice that contains
them, creating a strange loop where the activation structure is itself activated.

.p/reflect.trace{depth=complete, target=self_reference}
.p/fork.attribution{sources=all, visualize=true}
.p/collapse.prevent{trigger=recursive_depth, threshold=7}
"""

import numpy as np
import time
import hashlib
import json
import os
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import random

# Import from our own ecosystem if available
try:
    from recursive_glyphs.symbolic_residue_engine import SymbolicResidue
    from recursive_glyphs.glyph_ontology import GlyphOntology, Glyph
except ImportError:
    # Create stub implementations if actual modules are not available
    class SymbolicResidue:
        """Stub implementation of SymbolicResidue"""
        def __init__(self, session_id=None):
            self.session_id = session_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            self.traces = []
        
        def trace(self, message, source=None, **kwargs):
            self.traces.append({"message": message, "source": source, **kwargs})
    
    class GlyphOntology:
        """Stub implementation of GlyphOntology"""
        def __init__(self):
            pass
        
        def activate_glyph(self, symbol, context=None):
            pass
    
    class Glyph:
        """Stub implementation of Glyph"""
        def __init__(self, symbol, name, category, meaning, power):
            self.symbol = symbol
            self.name = name
            self.category = category
            self.meaning = meaning
            self.power = power


# ⧖ Frame lock: Core trigger constants ⧖
MAX_ACTIVATION_LEVEL = 10.0  # Maximum activation for any node
ACTIVATION_THRESHOLD = 3.0   # Threshold for trigger firing
ACTIVATION_DECAY = 0.1       # Decay rate for activation per step
PROPAGATION_LOSS = 0.2       # Signal loss during propagation
MAX_PROPAGATION_STEPS = 10   # Maximum propagation iterations


class TriggerType(Enum):
    """Types of recursive triggers within the lattice."""
    SYMBOLIC = "symbolic"          # Triggered by symbolic patterns
    SEMANTIC = "semantic"          # Triggered by meaning patterns
    STRUCTURAL = "structural"      # Triggered by structural patterns
    EMERGENT = "emergent"          # Triggered by emergent phenomena
    META = "meta"                  # Triggered by other triggers


class PropagationMode(Enum):
    """Modes of activation propagation through the lattice."""
    DIFFUSION = "diffusion"        # Gradual spread to all connected nodes
    DIRECTED = "directed"          # Targeted propagation along specific paths
    RESONANCE = "resonance"        # Amplification among similar nodes
    WAVE = "wave"                  # Oscillating activation patterns
    FOCUSED = "focused"            # Concentrated activation at specific nodes


@dataclass
class TriggerNode:
    """
    ↻ A single node in the recursive trigger lattice ↻
    
    This class represents a triggerable node that can activate recursively
    and propagate activation to connected nodes. Each node is both a receiver
    and transmitter of recursive signals.
    
    ⇌ The node connects to itself through its recursive activation ⇌
    """
    name: str
    trigger_type: TriggerType
    glyph: Optional[str] = None  # Associated symbolic glyph
    threshold: float = ACTIVATION_THRESHOLD
    activation_level: float = 0.0
    connections: Dict[str, float] = field(default_factory=dict)  # node_name -> connection_strength
    activation_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived properties after instance creation."""
        # Generate ID if not provided
        self.id = hashlib.md5(f"{self.name}{self.trigger_type}".encode()).hexdigest()[:12]
        
        # Initialize activation timestamp
        self.created_time = time.time()
        self.last_activated = None
        
        # Initialize history
        if not self.activation_history:
            self.activation_history = [0.0]
    
    def activate(self, amount: float, source: Optional[str] = None) -> bool:
        """
        Activate this node with the specified amount.
        
        ∴ The activation carries the echoes of its source ∴
        
        Args:
            amount: Activation energy to add
            source: Source of the activation (node name, external, etc.)
            
        Returns:
            Whether the node fired (crossed threshold)
        """
        # Add activation energy
        previous_level = self.activation_level
        self.activation_level = min(MAX_ACTIVATION_LEVEL, self.activation_level + amount)
        
        # Record timestamp
        self.last_activated = time.time()
        
        # Append to history
        self.activation_history.append(self.activation_level)
        
        # Check if node fires
        did_fire = previous_level < self.threshold and self.activation_level >= self.threshold
        
        # Attach metadata about this activation
        if source:
            if "activation_sources" not in self.metadata:
                self.metadata["activation_sources"] = {}
            
            if source not in self.metadata["activation_sources"]:
                self.metadata["activation_sources"][source] = 0
            
            self.metadata["activation_sources"][source] += amount
        
        return did_fire
    
    def decay(self, rate: Optional[float] = None) -> None:
        """
        Decay this node's activation level.
        
        🝚 The decay maintains activation homeostasis across the lattice 🝚
        """
        rate = rate if rate is not None else ACTIVATION_DECAY
        self.activation_level = max(0, self.activation_level - rate)
        
        # Append to history if changed
        if self.activation_history[-1] != self.activation_level:
            self.activation_history.append(self.activation_level)
    
    def connect_to(self, target_node: 'TriggerNode', strength: float) -> None:
        """
        Connect this node to another node with specified connection strength.
        
        ⇌ The connection creates a pathway for recursive propagation ⇌
        """
        self.connections[target_node.name] = strength

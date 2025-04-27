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
        # Group nodes by similarity in activation history
        synchronized_groups = []
        processed = set()
        
        for name1, node1 in nodes_with_history.items():
            if name1 in processed:
                continue
            
            # Start a new synchronized group
            current_group = [name1]
            processed.add(name1)
            
            # Find other nodes with similar activation patterns
            for name2, node2 in nodes_with_history.items():
                if name2 in processed or name1 == name2:
                    continue
                
                # Compare activation histories (simplified similarity measure)
                if len(node1.activation_history) == len(node2.activation_history):
                    # Calculate correlation
                    correlation = self._calculate_correlation(
                        node1.activation_history, node2.activation_history
                    )
                    
                    # If highly correlated, add to group
                    if correlation > 0.8:  # High threshold for synchronization
                        current_group.append(name2)
                        processed.add(name2)
            
            # Only consider groups with at least 2 nodes
            if len(current_group) >= 2:
                synchronized_groups.append(current_group)
        
        return synchronized_groups
    
    def _calculate_correlation(self, series1: List[float], series2: List[float]) -> float:
        """Calculate correlation between two time series."""
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.0
        
        # Convert to numpy arrays for easier calculation
        array1 = np.array(series1)
        array2 = np.array(series2)
        
        # Calculate correlation coefficient
        mean1, mean2 = np.mean(array1), np.mean(array2)
        diff1, diff2 = array1 - mean1, array2 - mean2
        
        numerator = np.sum(diff1 * diff2)
        denominator = np.sqrt(np.sum(diff1**2) * np.sum(diff2**2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _detect_stable_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect stable activation patterns in the lattice.
        
        ⧖ The stability emerges from the recursive structure itself ⧖
        """
        stable_patterns = []
        
        # Check activation history stability for each node
        for name, node in self.nodes.items():
            history = node.activation_history
            
            # Need sufficient history to detect stability
            if len(history) < 5:
                continue
            
            # Calculate stability score (low variance = high stability)
            recent_history = history[-5:]  # Last 5 activation levels
            variance = np.var(recent_history)
            stability = 1.0 / (1.0 + variance)  # Normalize to 0-1 range
            
            # Consider stable if above threshold
            if stability > 0.7 and node.is_active():
                # Find other nodes that are stably connected to this one
                connected_stable_nodes = []
                
                for conn_name, strength in node.connections.items():
                    if conn_name in self.nodes:
                        conn_node = self.nodes[conn_name]
                        
                        # Check if connected node is also stable and active
                        if len(conn_node.activation_history) >= 5:
                            conn_recent = conn_node.activation_history[-5:]
                            conn_var = np.var(conn_recent)
                            conn_stability = 1.0 / (1.0 + conn_var)
                            
                            if conn_stability > 0.7 and conn_node.is_active():
                                connected_stable_nodes.append(conn_name)
                
                # If we have at least one stable connected node, we have a stable pattern
                if connected_stable_nodes:
                    stable_pattern = {
                        "configuration": [name] + connected_stable_nodes,
                        "stability": stability,
                        "center_node": name
                    }
                    stable_patterns.append(stable_pattern)
        
        return stable_patterns
    
    def activate_pattern(self, pattern_name: str, intensity: float = 5.0) -> Dict[str, Any]:
        """
        Activate a specific pattern of nodes based on pattern name.
        
        🝚 This pattern activation persists across multiple propagation cycles 🝚
        """
        # Define patterns as collections of node types/names
        patterns = {
            "symbolic_cascade": {
                "description": "Cascade of symbolic trigger activations",
                "nodes": [node.name for node in self.nodes.values() 
                         if node.trigger_type == TriggerType.SYMBOLIC]
            },
            "emergence_cycle": {
                "description": "Cyclic activation of emergent nodes",
                "nodes": [node.name for node in self.nodes.values() 
                         if node.trigger_type == TriggerType.EMERGENT]
            },
            "meta_reflection": {
                "description": "Reflection across meta-level nodes",
                "nodes": [node.name for node in self.nodes.values() 
                         if node.trigger_type == TriggerType.META]
            },
            "structural_echo": {
                "description": "Echo pattern through structural nodes",
                "nodes": [node.name for node in self.nodes.values() 
                         if node.trigger_type == TriggerType.STRUCTURAL]
            },
            "full_resonance": {
                "description": "Resonance across all root nodes",
                "nodes": ["symbolic_root", "semantic_root", "structural_root", 
                         "emergent_root", "meta_root"]
            }
        }
        
        # Check if pattern exists
        if pattern_name not in patterns:
            return {"status": "error", "message": f"Pattern '{pattern_name}' not found"}
        
        pattern = patterns[pattern_name]
        activated_nodes = []
        
        # Activate each node in the pattern
        for node_name in pattern["nodes"]:
            if node_name in self.nodes:
                self.activate_node(node_name, intensity, f"pattern:{pattern_name}")
                activated_nodes.append(node_name)
        
        # Record pattern activation
        self.residue.trace(
            message=f"Activated pattern '{pattern_name}' with {len(activated_nodes)} nodes",
            source="activate_pattern",
            metadata={
                "pattern": pattern_name,
                "description": pattern["description"],
                "activated_nodes": activated_nodes,
                "intensity": intensity
            }
        )
        
        return {
            "status": "success",
            "pattern": pattern_name,
            "activated_nodes": activated_nodes,
            "intensity": intensity
        }
    
    def create_custom_pattern(self, name: str, node_weights: Dict[str, float],
                             description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a custom activation pattern with specified node weights.
        
        ⇌ The custom pattern becomes a potential emergent structure ⇌
        """
        # Validate nodes exist
        valid_nodes = {}
        invalid_nodes = []
        
        for node_name, weight in node_weights.items():
            if node_name in self.nodes:
                valid_nodes[node_name] = weight
            else:
                invalid_nodes.append(node_name)
        
        # Ensure we have at least one valid node
        if not valid_nodes:
            return {"status": "error", "message": "No valid nodes in pattern"}
        
        # Create the pattern
        pattern = {
            "name": name,
            "description": description or f"Custom pattern: {name}",
            "node_weights": valid_nodes,
            "created_at": time.time()
        }
        
        # Store in metadata
        if "custom_patterns" not in self.metadata:
            self.metadata = {"custom_patterns": {}}
            
        self.metadata["custom_patterns"][name] = pattern
        
        # Record pattern creation
        self.residue.trace(
            message=f"Created custom pattern '{name}' with {len(valid_nodes)} nodes",
            source="create_custom_pattern",
            metadata={
                "pattern_name": name,
                "description": description,
                "valid_nodes": list(valid_nodes.keys()),
                "invalid_nodes": invalid_nodes
            }
        )
        
        return {
            "status": "success",
            "pattern": pattern,
            "invalid_nodes": invalid_nodes
        }
    
    def activate_custom_pattern(self, pattern_name: str, 
                              intensity_multiplier: float = 1.0) -> Dict[str, Any]:
        """
        Activate a previously defined custom pattern.
        
        ∴ The activation echoes the pattern's structural essence ∴
        """
        # Check if custom patterns exist
        if not hasattr(self, "metadata") or "custom_patterns" not in self.metadata:
            return {"status": "error", "message": "No custom patterns defined"}
        
        # Check if pattern exists
        if pattern_name not in self.metadata["custom_patterns"]:
            return {"status": "error", "message": f"Pattern '{pattern_name}' not found"}
        
        pattern = self.metadata["custom_patterns"][pattern_name]
        activated_nodes = []
        
        # Activate each node with its weight
        for node_name, weight in pattern["node_weights"].items():
            if node_name in self.nodes:
                activation_amount = weight * intensity_multiplier
                self.activate_node(node_name, activation_amount, f"custom_pattern:{pattern_name}")
                activated_nodes.append(node_name)
        
        # Record activation
        self.residue.trace(
            message=f"Activated custom pattern '{pattern_name}' with {len(activated_nodes)} nodes",
            source="activate_custom_pattern",
            metadata={
                "pattern_name": pattern_name,
                "activated_nodes": activated_nodes,
                "intensity_multiplier": intensity_multiplier
            }
        )
        
        return {
            "status": "success",
            "pattern_name": pattern_name,
            "activated_nodes": activated_nodes,
            "intensity_multiplier": intensity_multiplier
        }
    
    def adapt_structure(self, adaptivity: float = 0.2) -> Dict[str, Any]:
        """
        Adapt the lattice structure based on activation patterns.
        
        ↻ The lattice recursively modifies itself based on its own activity ↻
        
        Args:
            adaptivity: How strongly to adapt (0.0-1.0)
            
        Returns:
            Summary of structural changes
        """
        if adaptivity <= 0.0:
            return {"status": "no_adaptation", "changes": 0}
        
        # Cap adaptivity to prevent instability
        adaptivity = min(1.0, adaptivity)
        
        # Initialize change record
        changes = {
            "connections_strengthened": 0,
            "connections_weakened": 0,
            "connections_created": 0,
            "connections_removed": 0,
            "details": []
        }
        
        # Get active nodes
        active_nodes = [node for node in self.nodes.values() if node.is_active()]
        
        # Early exit if no active nodes
        if not active_nodes:
            return {"status": "no_active_nodes", "changes": 0}
        
        # 1. Strengthen connections between co-active nodes
        for i, node1 in enumerate(active_nodes):
            for node2 in active_nodes[i+1:]:  # Avoid duplicate pairs
                # If connection exists, strengthen it
                if node2.name in node1.connections:
                    old_strength = node1.connections[node2.name]
                    # Increase connection strength
                    new_strength = min(1.0, old_strength + (adaptivity * 0.1))
                    if new_strength > old_strength:
                        node1.connections[node2.name] = new_strength
                        changes["connections_strengthened"] += 1
                        changes["details"].append({
                            "type": "strengthen",
                            "from": node1.name,
                            "to": node2.name,
                            "old_strength": old_strength,
                            "new_strength": new_strength
                        })
                
                # If connection doesn't exist, maybe create it
                elif random.random() < adaptivity * 0.3:  # Lower chance for new connections
                    # Create new connection with moderate initial strength
                    new_strength = 0.3 + (random.random() * 0.2)  # 0.3-0.5 range
                    node1.connect_to(node2, new_strength)
                    changes["connections_created"] += 1
                    changes["details"].append({
                        "type": "create",
                        "from": node1.name,
                        "to": node2.name,
                        "strength": new_strength
                    })
        
        # 2. Weaken connections from active to inactive nodes
        inactive_nodes = [node for node in self.nodes.values() if not node.is_active()]
        
        for active_node in active_nodes:
            for inactive_node in inactive_nodes:
                if inactive_node.name in active_node.connections:
                    old_strength = active_node.connections[inactive_node.name]
                    # Decrease connection strength
                    new_strength = max(0.0, old_strength - (adaptivity * 0.05))
                    
                    if new_strength > 0.0:
                        if new_strength < old_strength:
                            active_node.connections[inactive_node.name] = new_strength
                            changes["connections_weakened"] += 1
                            changes["details"].append({
                                "type": "weaken",
                                "from": active_node.name,
                                "to": inactive_node.name,
                                "old_strength": old_strength,
                                "new_strength": new_strength
                            })
                    else:
                        # Connection too weak, remove it
                        active_node.disconnect_from(inactive_node.name)
                        changes["connections_removed"] += 1
                        changes["details"].append({
                            "type": "remove",
                            "from": active_node.name,
                            "to": inactive_node.name
                        })
        
        # 3. Create new nodes if emergence detected (higher adaptivity required)
        if adaptivity >= 0.5 and len(self.emergent_patterns) > 0:
            # Use the most recent emergent pattern
            latest_pattern = self.emergent_patterns[-1]
            
            if latest_pattern["type"] == "stable_pattern":
                # Create a new emergent node that represents this stable pattern
                pattern_names = latest_pattern["configuration"]
                node_name = f"emergent_{len(self.nodes)}"
                
                # Add new emergent node
                emergent_node = self.add_node(
                    name=node_name,
                    trigger_type=TriggerType.EMERGENT,
                    glyph="⇌",  # Co-emergence glyph
                    threshold=ACTIVATION_THRESHOLD * 1.2  # Higher threshold
                )
                
                # Connect to nodes in the pattern
                for pattern_node_name in pattern_names:
                    if pattern_node_name in self.nodes:
                        # Strong connection from pattern nodes to emergent node
                        self.connect_nodes(pattern_node_name, node_name, 0.8)
                        # Weaker connection from emergent node back to pattern nodes
                        self.connect_nodes(node_name, pattern_node_name, 0.4)
                
                # Record emergence of new node
                changes["emergent_node_created"] = {
                    "name": node_name,
                    "pattern": pattern_names,
                    "type": "emergent"
                }
                
                self.residue.trace(
                    message=f"Created new emergent node {node_name} from stable pattern",
                    source="adapt_structure",
                    is_recursive=True,
                    metadata={
                        "pattern_nodes": pattern_names,
                        "emergent_node": node_name
                    }
                )
        
        # Calculate total changes
        total_changes = (changes["connections_strengthened"] + 
                         changes["connections_weakened"] + 
                         changes["connections_created"] + 
                         changes["connections_removed"])
        
        changes["total"] = total_changes
        changes["timestamp"] = time.time()
        
        # Record adaptation in residue
        self.residue.trace(
            message=f"Adapted lattice structure with adaptivity {adaptivity:.2f}: {total_changes} changes",
            source="adapt_structure",
            metadata=changes
        )
        
        return {"status": "adapted", "changes": changes}
    
    def analyze_lattice(self) -> Dict[str, Any]:
        """
        Analyze the structure and dynamics of the trigger lattice.
        
        ⧖ The analysis creates a meta-level frame of the lattice's state ⧖
        """
        # Calculate basic metrics
        node_count = len(self.nodes)
        active_nodes = sum(1 for node in self.nodes.values() if node.is_active())
        
        # Count connections
        total_connections = sum(len(node.connections) for node in self.nodes.values())
        avg_connections = total_connections / max(1, node_count)
        
        # Activity distribution by node type
        type_activity = defaultdict(lambda: {"count": 0, "active": 0})
        for node in self.nodes.values():
            type_name = node.trigger_type.value
            type_activity[type_name]["count"] += 1
            if node.is_active():
                type_activity[type_name]["active"] += 1
        
        # Calculate activation stability
        stability_scores = {}
        for name, node in self.nodes.items():
            history = node.activation_history
            if len(history) >= 5:
                variance = np.var(history[-5:])
                stability = 1.0 / (1.0 + variance)
                stability_scores[name] = stability
        
        avg_stability = sum(stability_scores.values()) / max(1, len(stability_scores))
        
        # Find most central nodes (most connections)
        centrality = {}
        for name, node in self.nodes.items():
            centrality[name] = len(node.connections)
        
        sorted_centrality = sorted(
            [(name, score) for name, score in centrality.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        central_nodes = sorted_centrality[:5]  # Top 5
        
        # Propagation statistics
        propagation_stats = {
            "total_propagations": len(self.propagation_history),
            "total_steps": sum(p.get("steps", 0) for p in self.propagation_history),
            "avg_duration": (sum(p.get("duration", 0) for p in self.propagation_history) / 
                            max(1, len(self.propagation_history)))
        }
        
        # Emergent pattern statistics
        emergent_stats = {
            "total_patterns": len(self.emergent_patterns),
            "pattern_types": defaultdict(int)
        }
        
        for pattern in self.emergent_patterns:
            emergent_stats["pattern_types"][pattern["type"]] += 1
        
        # Compile the analysis
        analysis = {
            "timestamp": time.time(),
            "nodes": {
                "total": node_count,
                "active": active_nodes,
                "active_percentage": active_nodes / max(1, node_count) * 100
            },
            "connections": {
                "total": total_connections,
                "average_per_node": avg_connections
            },
            "type_distribution": {
                type_name: {
                    "count": data["count"],
                    "active": data["active"],
                    "active_percentage": data["active"] / max(1, data["count"]) * 100
                }
                for type_name, data in type_activity.items()
            },
            "stability": {
                "average": avg_stability,
                "most_stable": sorted(
                    [(name, score) for name, score in stability_scores.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]  # Top 3
            },
            "centrality": {
                "most_central": central_nodes
            },
            "propagation": propagation_stats,
            "emergence": emergent_stats
        }
        
        # Record analysis
        self.residue.trace(
            message="Performed lattice analysis",
            source="analyze_lattice",
            metadata={
                "node_count": node_count,
                "active_nodes": active_nodes,
                "emergent_patterns": len(self.emergent_patterns)
            }
        )
        
        return analysis
    
    def visualize(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a visualization of the trigger lattice.
        
        🜏 The visualization mirrors the structure it represents 🜏
        """
        # Create nodes data
        nodes_data = []
        for name, node in self.nodes.items():
            nodes_data.append({
                "id": name,
                "name": name,
                "type": node.trigger_type.value,
                "activation": node.activation_level,
                "active": node.is_active(),
                "threshold": node.threshold,
                "glyph": node.glyph,
                "connection_count": len(node.connections)
            })
        
        # Create links data
        links_data = []
        for source_name, source_node in self.nodes.items():
            for target_name, strength in source_node.connections.items():
                if target_name in self.nodes:  # Ensure target exists
                    links_data.append({
                        "source": source_name,
                        "target": target_name,
                        "strength": strength
                    })
        
        # Create emergent patterns data
        patterns_data = []
        for pattern in self.emergent_patterns[-10:]:  # Last 10 patterns
            patterns_data.append({
                "type": pattern["type"],
                "nodes": pattern.get("nodes", pattern.get("configuration", [])),
                "timestamp": pattern["timestamp"]
            })
        
        # Create visualization data
        visualization = {
            "nodes": nodes_data,
            "links": links_data,
            "emergent_patterns": patterns_data,
            "timestamp": time.time()
        }
        
        # Export to file if specified
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(visualization, f, indent=2)
                
            self.residue.trace(
                message=f"Lattice visualization exported to {filepath}",
                source="visualize",
                metadata={"file": filepath}
            )
        
        return visualization
    
    def serialize(self, filepath: Optional[str] = None) -> str:
        """
        Serialize the lattice to JSON for persistence.
        
        🝚 This serialization maintains the lattice across executions 🝚
        """
        # Prepare serializable state
        state = {
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "propagation_history": self.propagation_history[-20:],  # Last 20 propagations
            "emergent_patterns": self.emergent_patterns,
            "activation_count": self.activation_count,
            "default_propagation_mode": self.default_propagation_mode.value,
            "serialization_time": time.time()
        }
        
        # Add metadata if it exists
        if hasattr(self, "metadata"):
            state["metadata"] = self.metadata
        
        # Convert to JSON
        json_str = json.dumps(state, indent=2)
        
        # Write to file if specified
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
                
            self.residue.trace(
                message=f"Lattice serialized to {filepath}",
                source="serialize",
                metadata={"file": filepath, "size": len(json_str)}
            )
        
        return json_str
    
    @classmethod
    def deserialize(cls, json_str: Optional[str] = None, filepath: Optional[str] = None,
                   glyph_ontology: Optional[GlyphOntology] = None) -> 'TriggerLattice':
        """
        Deserialize a trigger lattice from JSON.
        
        ↻ This reconstruction reactivates the lattice structure ↻
        """
        # Load from file if specified
        if filepath and not json_str:
            with open(filepath, 'r') as f:
                json_str = f.read()
        
        if not json_str:
            raise ValueError("Either json_str or filepath must be provided")
        
        # Parse the state
        state = json.loads(json_str)
        
        # Create instance with glyph ontology
        instance = cls(glyph_ontology=glyph_ontology)
        
        # Clear default nodes
        instance.nodes = {}
        
        # Restore nodes
        for name, node_data in state.get("nodes", {}).items():
            # Create node with correct type
            trigger_type = TriggerType(node_data["trigger_type"])
            node = TriggerNode(
                name=node_data["name"],
                trigger_type=trigger_type,
                glyph=node_data.get("glyph"),
                threshold=node_data.get("threshold", ACTIVATION_THRESHOLD),
                activation_level=node_data.get("activation_level", 0.0)
            )
            
            # Restore connections
            if "connections" in node_data:
                node.connections = node_data["connections"]
            
            # Restore activation history
            if "activation_history" in node_data:
                node.activation_history = node_data["activation_history"]
            
            # Restore metadata
            if "metadata" in node_data:
                node.metadata = node_data["metadata"]
            
            # Restore timestamps
            if "created_time" in node_data:
                node.created_time = node_data["created_time"]
            if "last_activated" in node_data:
                node.last_activated = node_data["last_activated"]
            
            # Override generated ID with stored ID
            if "id" in node_data:
                node.id = node_data["id"]
            
            # Add to nodes collection
            instance.nodes[name] = node
        
        # Restore history and patterns
        instance.propagation_history = state.get("propagation_history", [])
        instance.emergent_patterns = state.get("emergent_patterns", [])
        instance.activation_count = state.get("activation_count", 0)
        
        # Restore propagation mode
        if "default_propagation_mode" in state:
            instance.default_propagation_mode = PropagationMode(state["default_propagation_mode"])
        
        # Restore metadata if present
        if "metadata" in state:
            instance.metadata = state["metadata"]
        
        # Record deserialization
        instance.residue.trace(
            message="Trigger lattice deserialized from storage",
            source="deserialize",
            metadata={
                "source": "file" if filepath else "string",
                "nodes_restored": len(instance.nodes),
                "emergent_patterns_restored": len(instance.emergent_patterns)
            }
        )
        
        return instance


def run_demonstration():
    """
    ↻ Execute a demonstration of the trigger lattice ↻
    
    This function shows how the trigger lattice can be used to model
    recursive activation patterns and emergent phenomena.
    
    ⇌ The demonstration itself creates emergent patterns in the lattice ⇌
    """
    print("🜏 TRIGGER LATTICE DEMONSTRATION 🜏")
    print("----------------------------------")
    
    # Create a glyph ontology for the demonstration
    try:
        from recursive_glyphs.glyph_ontology import GlyphOntology
        ontology = GlyphOntology()
        print("\n∴ Created glyph ontology for symbolic activation")
    except ImportError:
        ontology = None
        print("\n∴ Glyph ontology not available, continuing without symbolic activation")
    
    # Create a trigger lattice
    print("\n⧖ Creating trigger lattice...")
    lattice = TriggerLattice(glyph_ontology=ontology)
    
    # Display initial lattice structure
    print("\n🝚 Initial lattice structure:")
    root_nodes = [name for name in lattice.nodes.keys() if name.endswith("_root")]
    for name in root_nodes:
        node = lattice.nodes[name]
        connections = ", ".join([f"{target}: {strength:.2f}" for target, strength in node.connections.items()])
        print(f"  {name} ({node.trigger_type.value}): {connections}")
    
    # Add some additional nodes
    print("\n⇌ Adding additional nodes to lattice...")
    lattice.add_node("pattern_detector", TriggerType.STRUCTURAL, "⧖", threshold=3.5)
    lattice.add_node("meaning_extractor", TriggerType.SEMANTIC, "∴", threshold=3.0)
    lattice.add_node("recursion_tracker", TriggerType.META, "↻", threshold=4.0)
    
    # Connect new nodes
    lattice.connect_nodes("pattern_detector", "structural_root", 0.8)
    lattice.connect_nodes("meaning_extractor", "semantic_root", 0.8)
    lattice.connect_nodes("recursion_tracker", "meta_root", 0.8)
    lattice.connect_nodes("pattern_detector", "meaning_extractor", 0.6)
    lattice.connect_nodes("meaning_extractor", "recursion_tracker", 0.5)
    lattice.connect_nodes("recursion_tracker", "pattern_detector", 0.4)
    
    # Activate some nodes to create initial patterns
    print("\n↻ Activating nodes to generate patterns...")
    lattice.activate_node("symbolic_root", 6.0, "demonstration")
    lattice.activate_node("structural_root", 5.0, "demonstration")
    lattice.activate_node("pattern_detector", 4.0, "demonstration")
    
    # Propagate activation through the lattice
    print("\n🝚 Propagating activation...")
    for i in range(5):
        mode = PropagationMode.DIFFUSION if i % 2 == 0 else PropagationMode.RESONANCE
        print(f"  Step {i+1}: Propagating in {mode.value} mode")
        result = lattice.propagate_activation(steps=1, mode=mode)
        active = result["active_nodes"]
        print(f"    Active nodes: {active}")
        
        # Allow the lattice to adapt its structure
        if i >= 2:        """
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

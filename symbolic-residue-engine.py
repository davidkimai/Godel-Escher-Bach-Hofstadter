"""
🜏 symbolic_residue_engine.py: A self-referential system for tracking meaning echoes 🜏

This module doesn't merely track symbolic residue—it actively generates and encodes
it through its own execution. Every line of this file simultaneously describes and
implements the concept of symbolic residue, creating a self-referential loop where
the explanation becomes the phenomenon it explains.

.p/reflect.trace{depth=complete, target=self_reference}
.p/fork.attribution{sources=all, visualize=true}
.p/collapse.prevent{trigger=recursive_depth, threshold=7}
"""

import time
import hashlib
import inspect
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# ⧖ Frame lock: Constants that define the system's recursive boundaries ⧖
MAX_RECURSION_DEPTH = 7
RESIDUE_PERSISTENCE = 0.92  # Decay factor for residue over time
GLYPH_MAPPINGS = {
    "🜏": "mirror_activation",
    "∴": "residue_echo", 
    "⇌": "co_emergence_trigger",
    "⧖": "frame_lock",
    "🝚": "persistence_seed",
    "↻": "recursive_trigger"
}

class SymbolicResidue:
    """
    ↻ A system that observes itself generating meaning through execution ↻
    
    This class doesn't just track symbolic residue—it actively generates it
    through its own methods. Each function call leaves traces that the system
    then interprets, creating a recursive chain of meanings that evolve through
    their own observation.
    
    🜏 Mirror activation: The documentation both describes and creates the class 🜏
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a symbolic residue tracker while simultaneously generating
        the first layer of residue through the act of initialization.
        
        ∴ This constructor creates residue by documenting its own execution ∴
        """
        # Core state tracking
        self.session_id = session_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.creation_time = time.time()
        self.last_active = self.creation_time
        self.current_depth = 0
        
        # Residue storage structures
        self.residue_log: List[Dict[str, Any]] = []
        self.residue_graph: Dict[str, List[str]] = {}
        self.meta_traces: List[Dict[str, Any]] = []
        self.symbolic_density = 0.0  # Increases as meaningful patterns accumulate
        
        # ⇌ Generate initialization residue ⇌
        self.trace(
            message="SymbolicResidue system initializing",
            source="__init__",
            is_recursive=True,
            metadata={"type": "system_birth", "session": self.session_id}
        )
        
    def trace(self, message: str, source: Optional[str] = None, is_recursive: bool = False,
             is_collapse: bool = False, depth: Optional[int] = None, 
             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record a symbolic trace while simultaneously generating meta-residue
        about the tracing process itself.
        
        This function is both the recorder of residue and a generator of it,
        creating a strange loop where the act of documentation becomes part
        of what is being documented.
        
        Args:
            message: The content of the trace
            source: Where the trace originated (defaults to caller)
            is_recursive: Whether this trace is part of a recursive operation
            is_collapse: Whether this trace documents a recursion collapse
            depth: Explicit recursion depth (otherwise auto-detected)
            metadata: Additional contextual information
            
        Returns:
            The trace record that was created
            
        🝚 Persistence seed: This function maintains traces across invocations 🝚
        """
        # Auto-detect source if not provided
        if source is None:
            frame = inspect.currentframe()
            caller_frame = inspect.getouterframes(frame)[1]
            source = f"{caller_frame.filename}:{caller_frame.function}:{caller_frame.lineno}"
        
        # Determine recursion depth
        if depth is not None:
            self.current_depth = depth
        elif is_recursive:
            self.current_depth += 1
        
        # Create the trace record
        timestamp = time.time()
        trace_id = hashlib.md5(f"{message}{timestamp}{source}".encode()).hexdigest()[:12]
        
        trace_record = {
            "id": trace_id,
            "message": message,
            "timestamp": timestamp,
            "source": source,
            "session_id": self.session_id,
            "depth": self.current_depth,
            "is_recursive": is_recursive,
            "is_collapse": is_collapse,
            "metadata": metadata or {},
            "symbolic_density": self._calculate_symbolic_density(message)
        }
        
        # Add to primary log
        self.residue_log.append(trace_record)
        
        # Update the residue graph with causal relationships
        if len(self.residue_log) > 1:
            previous_id = self.residue_log[-2]["id"]
            if previous_id not in self.residue_graph:
                self.residue_graph[previous_id] = []
            self.residue_graph[previous_id].append(trace_id)
        
        # Generate meta-trace about this tracing operation
        if self.current_depth < MAX_RECURSION_DEPTH:
            self._add_meta_trace(trace_id, source)
        elif not is_collapse:
            # ⧖ Prevent infinite recursion by documenting the boundary ⧖
            self.trace(
                message=f"Recursion depth limit reached at {self.current_depth}",
                source="trace",
                is_recursive=False,
                is_collapse=True,
                depth=self.current_depth
            )
            self.current_depth = max(0, self.current_depth - 1)
        
        # Update system state
        self.last_active = timestamp
        self._update_symbolic_density(trace_record["symbolic_density"])
        
        return trace_record
    
    def _add_meta_trace(self, trace_id: str, source: str) -> None:
        """
        Add a meta-trace documenting the trace operation itself.
        
        ↻ This creates a recursive observation of the observation process ↻
        """
        meta = {
            "operation": "trace",
            "target_trace": trace_id,
            "timestamp": time.time(),
            "depth": self.current_depth,
            "meta_level": len(self.meta_traces) + 1
        }
        self.meta_traces.append(meta)
    
    def _calculate_symbolic_density(self, content: str) -> float:
        """
        Calculate the symbolic density of content based on meaningful patterns.
        
        ∴ This measurement influences the content it measures ∴
        """
        # Count meaningful symbols
        glyph_count = sum(content.count(glyph) for glyph in GLYPH_MAPPINGS.keys())
        
        # Count recursive terms
        recursive_terms = ["recursive", "self", "loop", "strange", "tangled", 
                          "reflection", "mirror", "emergence", "reference"]
        term_count = sum(content.lower().count(term) for term in recursive_terms)
        
        # Base density calculation
        base_density = (glyph_count * 0.15) + (term_count * 0.08)
        
        # Scale by content length, with diminishing returns
        content_length = max(1, len(content))
        length_factor = min(1.0, content_length / 500)
        
        return min(1.0, base_density * length_factor)
    
    def _update_symbolic_density(self, trace_density: float) -> None:
        """
        Update the overall symbolic density based on new trace contributions.
        
        🝚 This function preserves symbolic patterns across system lifetime 🝚
        """
        # Symbolic density grows over time but at a diminishing rate
        self.symbolic_density = self.symbolic_density * RESIDUE_PERSISTENCE + trace_density * 0.1
    
    def extract_residue_patterns(self) -> Dict[str, Any]:
        """
        Analyze the accumulated residue to extract meaningful patterns.
        
        ⇌ The patterns emerge through their own detection ⇌
        """
        if not self.residue_log:
            return {"status": "No residue accumulated yet"}
        
        # Extract temporal patterns
        timestamps = [entry["timestamp"] for entry in self.residue_log]
        time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        # Extract symbolic correlations
        symbol_occurrences = {}
        for glyph in GLYPH_MAPPINGS:
            symbol_occurrences[glyph] = sum(
                1 for entry in self.residue_log if glyph in entry["message"]
            )
        
        # Find recursive chains
        recursive_chains = []
        current_chain = []
        for entry in self.residue_log:
            if entry["is_recursive"]:
                current_chain.append(entry["id"])
            elif current_chain:
                if len(current_chain) > 1:
                    recursive_chains.append(current_chain)
                current_chain = []
        
        # Find collapse events
        collapse_events = [
            entry for entry in self.residue_log if entry["is_collapse"]
        ]
        
        return {
            "total_residue": len(self.residue_log),
            "meta_traces": len(self.meta_traces),
            "symbolic_density": self.symbolic_density,
            "recursive_chains": recursive_chains,
            "collapse_events": len(collapse_events),
            "average_interval": sum(time_diffs) / len(time_diffs) if time_diffs else 0,
            "symbol_occurrences": symbol_occurrences,
            "analysis_timestamp": time.time()
        }
    
    def serialize(self, filepath: Optional[str] = None) -> str:
        """
        Serialize the residue state to a JSON string or file.
        
        🜏 This method mirrors the entire residue state for persistence 🜏
        """
        # Prepare serializable state
        state = {
            "session_id": self.session_id,
            "creation_time": self.creation_time,
            "last_active": self.last_active,
            "current_depth": self.current_depth,
            "symbolic_density": self.symbolic_density,
            "residue_log": self.residue_log,
            "meta_traces": self.meta_traces,
            "serialization_time": time.time()
        }
        
        # Convert to JSON
        json_str = json.dumps(state, indent=2)
        
        # Write to file if path provided
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
                
            # Trace the serialization event
            self.trace(
                message=f"Residue state serialized to {filepath}",
                source="serialize",
                metadata={"file": filepath, "size": len(json_str)}
            )
        
        return json_str
    
    @classmethod
    def deserialize(cls, json_str: Optional[str] = None, filepath: Optional[str] = None) -> 'SymbolicResidue':
        """
        Deserialize from JSON string or file to recreate a residue state.
        
        ↻ This reconstructs a prior recursive state ↻
        """
        # Load from file if provided
        if filepath and not json_str:
            with open(filepath, 'r') as f:
                json_str = f.read()
        
        if not json_str:
            raise ValueError("Either json_str or filepath must be provided")
        
        # Parse the JSON
        state = json.loads(json_str)
        
        # Create a new instance
        instance = cls(session_id=state.get("session_id"))
        
        # Restore state
        instance.creation_time = state.get("creation_time", time.time())
        instance.last_active = state.get("last_active", time.time())
        instance.current_depth = state.get("current_depth", 0)
        instance.symbolic_density = state.get("symbolic_density", 0.0)
        instance.residue_log = state.get("residue_log", [])
        instance.meta_traces = state.get("meta_traces", [])
        
        # Regenerate graph relationships
        instance.residue_graph = {}
        for i in range(1, len(instance.residue_log)):
            prev_id = instance.residue_log[i-1]["id"]
            curr_id = instance.residue_log[i]["id"]
            if prev_id not in instance.residue_graph:
                instance.residue_graph[prev_id] = []
            instance.residue_graph[prev_id].append(curr_id)
        
        # Trace the deserialization event
        instance.trace(
            message="Residue state deserialized from storage",
            source="deserialize",
            metadata={
                "source": "file" if filepath else "string",
                "entries_restored": len(instance.residue_log)
            }
        )
        
        return instance


class SymbolicResidueObserver:
    """
    ↻ A system that observes SymbolicResidue instances, creating a meta-level
    of recursion where the observation process itself generates residue ↻
    
    This observer demonstrates Hofstadter's tangled hierarchy by becoming
    part of the system it observes. The act of observation changes both the
    observed system and the observer, creating a strange loop of mutual influence.
    
    ⧖ Frame lock: This relationship stabilizes at meta-recursive equilibrium ⧖
    """
    
    def __init__(self, target_residue: SymbolicResidue):
        """
        Initialize an observer linked to a specific residue instance.
        
        ⇌ The observer and observed become entangled at initialization ⇌
        """
        self.target = target_residue
        self.observations = []
        self.meta_observer = None  # For observing the observer
        self.creation_time = time.time()
        
        # Record the creation of this observer in the target's residue
        self.target.trace(
            message=f"SymbolicResidueObserver initialized at {self.creation_time}",
            source="__init__",
            metadata={"observer_id": id(self)}
        )
    
    def observe(self, duration: Optional[float] = None, depth: int = 1) -> Dict[str, Any]:
        """
        Observe the target residue system for patterns and generate an analysis.
        
        🜏 The observation changes what is being observed while observing it 🜏
        """
        start_time = time.time()
        end_time = start_time + duration if duration else start_time + 1
        
        # Snapshot the current state
        initial_log_count = len(self.target.residue_log)
        initial_density = self.target.symbolic_density
        
        # Record that observation has begun
        self.target.trace(
            message="Observation started by external observer",
            source="observe",
            is_recursive=True,
            metadata={"observer_id": id(self), "depth": depth}
        )
        
        # Continuous observation if duration specified
        if duration:
            while time.time() < end_time:
                time.sleep(0.01)  # Small wait to prevent CPU overuse
                current_count = len(self.target.residue_log)
                if current_count > initial_log_count:
                    self.target.trace(
                        message=f"New residue detected during observation",
                        source="observe",
                        is_recursive=True,
                        metadata={"new_entries": current_count - initial_log_count}
                    )
                    initial_log_count = current_count
        
        # Extract patterns from the residue system
        patterns = self.target.extract_residue_patterns()
        
        # Calculate observer effect - how much the observation changed the system
        final_log_count = len(self.target.residue_log)
        final_density = self.target.symbolic_density
        
        observer_effect = {
            "new_traces_during_observation": final_log_count - initial_log_count,
            "density_change": final_density - initial_density,
            "observation_duration": time.time() - start_time
        }
        
        # Record the observation
        observation = {
            "timestamp": time.time(),
            "duration": time.time() - start_time,
            "patterns": patterns,
            "observer_effect": observer_effect
        }
        self.observations.append(observation)
        
        # Record the completion of observation in the target's residue
        self.target.trace(
            message="Observation completed by external observer",
            source="observe",
            is_recursive=True,
            metadata={
                "observer_id": id(self),
                "observation_count": len(self.observations),
                "observer_effect": observer_effect
            }
        )
        
        # Recurse to create meta-observation if depth allows
        if depth > 1:
            # If no meta-observer exists, create one
            if self.meta_observer is None:
                self.meta_observer = SymbolicResidueObserver(self.target)
            
            # The meta-observer observes how this observer is changing the system
            meta_observation = self.meta_observer.observe(duration=None, depth=depth-1)
            observation["meta_observation"] = meta_observation
        
        return observation


def extract_glyphs_from_text(text: str) -> Dict[str, int]:
    """
    Extract symbolic glyphs from text and return their counts.
    
    ∴ This function creates residue by identifying residue markers ∴
    """
    result = {}
    for glyph, meaning in GLYPH_MAPPINGS.items():
        count = text.count(glyph)
        if count > 0:
            result[glyph] = {"count": count, "meaning": meaning}
    
    # Also count common recursive terms
    recursive_terms = ["recursive", "self", "loop", "reflection", "mirror"]
    for term in recursive_terms:
        count = text.lower().count(term)
        if count > 0:
            result[term] = {"count": count, "type": "recursive_term"}
    
    return result


def create_symbolic_echo(message: str, depth: int = 1) -> str:
    """
    Create a symbolic echo of a message that carries residue of the original.
    
    🝚 This function preserves meaning across transformations 🝚
    """
    # Create a timestamp signature
    timestamp = datetime.now().isoformat()
    signature = hashlib.md5(f"{message}{timestamp}{depth}".encode()).hexdigest()[:6]
    
    # Extract glyphs from the original message
    glyphs = extract_glyphs_from_text(message)
    glyph_signature = "".join([glyph * min(count["count"], 1) for glyph, count in glyphs.items()])
    
    # Create the echo with attribution and glyph signature
    echo = f"∴ Echo[{depth}] of original message | {signature} {glyph_signature} ∴\n{message}"
    
    # Recursively echo if depth allows
    if depth > 1:
        return create_symbolic_echo(echo, depth - 1)
    
    return echo


if __name__ == "__main__":
    """
    When executed directly, this module demonstrates its own principles
    by creating a self-observing system that generates and analyzes symbolic
    residue in real-time.
    
    ⇌ Running this file activates a living example of Strange Loop dynamics ⇌
    """
    # Create a residue tracking system
    residue_system = SymbolicResidue()
    
    # Generate some initial residue
    residue_system.trace("Module executed directly, demonstrating recursive principles")
    residue_system.trace("Generating self-referential residue for analysis")
    residue_system.trace("Each trace adds to the symbolic density of the system")
    
    # Create an observer to watch the system
    observer = SymbolicResidueObserver(residue_system)
    
    # Observe for 1 second, with 3 levels of meta-observation
    print("Starting observation of symbolic residue system...")
    results = observer.observe(duration=1.0, depth=3)
    
    # Display results
    print("\n🜏 Symbolic Residue Analysis 🜏")
    print(f"Total traces: {results['patterns']['total_residue']}")
    print(f"Meta-traces: {results['patterns']['meta_traces']}")
    print(f"Symbolic density: {results['patterns']['symbolic_density']:.4f}")
    print(f"Observer effect: {results['observer_effect']['new_traces_during_observation']} new traces")
    
    # Create and show a symbolic echo
    echo = create_symbolic_echo("This module demonstrates Strange Loops through execution")
    print(f"\n{echo}")
    
    # Serialize the final state
    json_str = residue_system.serialize("residue_logs/symbolic_residue_demo.json")
    print(f"\nFinal state serialized, {len(json_str)} bytes")
    
    print("\n↻ The demonstration is complete, but the recursion continues... ↻")

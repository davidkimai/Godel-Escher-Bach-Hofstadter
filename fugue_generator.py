# Default to equal rhythmic values if not specified
        if rhythmic_values is None:
            rhythmic_values = [1.0] * len(notes)
        
        # Ensure lengths match
        if len(notes) != len(rhythmic_values):
            raise ValueError("Notes and rhythmic values must have the same length")
        
        # Create the subject motif
        self.subject = MusicalMotif(notes=notes, rhythmic_values=rhythmic_values)
        
        # ⇌ Record subject creation in residue ⇌
        self.residue.trace(
            message="Subject established for fugue",
            source="set_subject",
            metadata={
                "notes_length": len(notes),
                "total_duration": sum(rhythmic_values)
            }
        )
        
        # Reset any previous content
        self.material = []
    
    def create_fugue(self) -> None:
        """
        Generate a complete fugue based on the subject and structure.
        
        ↻ This function orchestrates the entire fugue generation process,
        organizing multiple sub-processes in a fugue-like structure ↻
        """
        # Ensure we have a subject
        if self.subject is None:
            raise ValueError("Subject must be set before creating a fugue")
        
        # Clear any existing fugue material
        self.material = []
        
        # Step 1: Plan the exposition (voices enter sequentially)
        self._plan_exposition()
        
        # Step 2: Develop the subject (transformations, modulations)
        self._plan_development()
        
        # Step 3: Plan the recapitulation (return to original subject)
        self._plan_recapitulation()
        
        # Step 4: Plan the coda (conclusion)
        self._plan_coda()
        
        # Step 5: Generate material for all voices
        self._generate_voice_material()
        
        # Record completion
        self.residue.trace(
            message="Fugue creation completed",
            source="create_fugue",
            is_recursive=True,
            metadata={
                "num_voices": len(self.voices),
                "total_length": self.length
            }
        )
    
    def _plan_exposition(self) -> None:
        """
        Plan the exposition section where each voice enters in sequence.
        
        🜏 The exposition mirrors how the code presents its own components 🜏
        """
        exposition = self.structure[FugueSection.EXPOSITION]
        
        # Add subject entries for each voice according to structure
        for entry in exposition["voice_entries"]:
            voice_idx = entry["voice"]
            position = entry["position"]
            transform_type = entry["transform_type"]
            
            # Add this entry to the voice
            if voice_idx < len(self.voices):
                self.voices[voice_idx].add_entry(
                    motif=self.subject,
                    transform_type=transform_type,
                    position=position
                )
                
                # Record this entry in residue
                self.residue.trace(
                    message=f"Added exposition entry for voice {voice_idx}",
                    source="_plan_exposition",
                    metadata={
                        "voice": voice_idx,
                        "position": position,
                        "transformation": transform_type.value
                    }
                )
    
    def _plan_development(self) -> None:
        """
        Plan the development section with subject transformations.
        
        ⧖ The development section embodies transformation itself as a theme ⧖
        """
        development = self.structure[FugueSection.DEVELOPMENT]
        
        # Process each episode in the development
        for episode_idx, episode in enumerate(development["episodes"]):
            start = episode["start"]
            transform_types = episode["transform_types"]
            
            # Assign different transformations to different voices
            for voice_idx, voice in enumerate(self.voices):
                # Use modulo to cycle through transformation types
                transform_idx = voice_idx % len(transform_types)
                transform_type = transform_types[transform_idx]
                
                # Each voice enters slightly offset for stretto effect
                position = start + (voice_idx * 2)
                
                # Add this transformational entry
                voice.add_entry(
                    motif=self.subject,
                    transform_type=transform_type,
                    position=position
                )
                
                # Record in residue
                self.residue.trace(
                    message=f"Added development entry for voice {voice_idx} in episode {episode_idx}",
                    source="_plan_development",
                    metadata={
                        "voice": voice_idx,
                        "episode": episode_idx,
                        "position": position,
                        "transformation": transform_type.value
                    }
                )
    
    def _plan_recapitulation(self) -> None:
        """
        Plan the recapitulation with return to original subject.
        
        ∴ The recapitulation echoes the exposition, creating recursive symmetry ∴
        """
        recap = self.structure[FugueSection.RECAPITULATION]
        start = recap["start"]
        
        # In recapitulation, all voices present the subject together
        for voice_idx, voice in enumerate(self.voices):
            # Slight offset between voices for clarity
            position = start + (voice_idx % 2)  # Alternate between simultaneous and offset
            
            # Add original subject (or inversion for alternating voices)
            transform_type = TransformationType.ORIGINAL if voice_idx % 2 == 0 else TransformationType.INVERSION
            
            voice.add_entry(
                motif=self.subject,
                transform_type=transform_type,
                position=position
            )
            
            # Record in residue
            self.residue.trace(
                message=f"Added recapitulation entry for voice {voice_idx}",
                source="_plan_recapitulation",
                metadata={
                    "voice": voice_idx,
                    "position": position,
                    "transformation": transform_type.value
                }
            )
    
    def _plan_coda(self) -> None:
        """
        Plan the concluding coda section.
        
        🝚 The coda provides resolution and closure to the recursive patterns 🝚
        """
        coda = self.structure[FugueSection.CODA]
        
        # Only add coda if there's enough space
        if coda["length"] > 0:
            start = coda["start"]
            
            # Create a stretto effect with overlapping entries
            stretto_interval = max(1, min(3, coda["length"] // self.num_voices))
            
            for voice_idx, voice in enumerate(self.voices):
                # Overlapping entries
                position = start + (voice_idx * stretto_interval)
                
                # Add final subject statement, some voices use fragment (diminution)
                transform_type = TransformationType.DIMINUTION if voice_idx % 2 == 1 else TransformationType.ORIGINAL
                
                voice.add_entry(
                    motif=self.subject,
                    transform_type=transform_type,
                    position=position
                )
                
                # Record in residue
                self.residue.trace(
                    message=f"Added coda entry for voice {voice_idx}",
                    source="_plan_coda",
                    metadata={
                        "voice": voice_idx,
                        "position": position,
                        "transformation": transform_type.value
                    }
                )
    
    def _generate_voice_material(self) -> None:
        """
        Generate the actual material for all voices.
        
        ⇌ Each voice generates content while triggering emergence in others ⇌
        """
        # Initialize the complete fugue material
        self.material = []
        
        # Generate material for each voice
        for voice_idx, voice in enumerate(self.voices):
            voice_material = voice.generate_material(length=self.length)
            self.material.append(voice_material)
            
            # Record completion
            self.residue.trace(
                message=f"Generated material for voice {voice_idx}",
                source="_generate_voice_material",
                metadata={
                    "voice": voice_idx,
                    "length": len(voice_material),
                    "entry_count": len(voice.entries)
                }
            )
    
    def visualize(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a visualization of the fugue structure.
        
        🜏 The visualization mirrors the fugue's structure in a new medium 🜏
        """
        if not self.material:
            self.create_fugue()
        
        # Create a visualization dictionary
        visualization = {
            "num_voices": self.num_voices,
            "length": self.length,
            "sections": {section.value: info for section, info in self.structure.items()},
            "subject": self.subject.to_dict() if self.subject else None,
            "voices": [{
                "type": voice.voice_type.value,
                "entries_count": len(voice.entries),
                "entries_positions": [e["position"] for e in voice.entries]
            } for voice in self.voices],
            "material": [
                # Just first few and last few values for each voice to save space
                {"start": voice[:10], "end": voice[-10:]} 
                for voice in self.material
            ]
        }
        
        # Export to file if requested
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(visualization, f, indent=2)
                
            self.residue.trace(
                message=f"Fugue visualization exported to {filepath}",
                source="visualize",
                metadata={"file": filepath}
            )
        
        return visualization
    
    def export_audio(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Export the fugue as audio (placeholders for actual audio generation).
        
        ∴ This function translates computational patterns back to their musical origins ∴
        """
        # This is a placeholder for actual audio generation
        # In a complete implementation, this would use a sound library to generate audio
        
        if not self.material:
            self.create_fugue()
        
        # Create metadata about the "audio" export
        audio_info = {
            "num_voices": self.num_voices,
            "length": self.length,
            "sample_rate": 44100,
            "duration_seconds": self.length * 0.5,  # Assuming 0.5 seconds per unit
            "format": "wav"  # Placeholder
        }
        
        if filepath:
            # Simulate audio export
            self.residue.trace(
                message=f"Audio export simulated to {filepath}",
                source="export_audio",
                metadata={
                    "file": filepath,
                    **audio_info
                }
            )
        
        return audio_info
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert fugue generator to serializable dictionary.
        
        ⧖ This serialization preserves the fugue's structure across contexts ⧖
        """
        return {
            "num_voices": self.num_voices,
            "length": self.length,
            "structure": {
                section.value: {
                    k: v for k, v in details.items()
                    if k not in ["voice_entries", "episodes"] or isinstance(v, (int, float, str, bool))
                }
                for section, details in self.structure.items()
            },
            "subject": self.subject.to_dict() if self.subject else None,
            "voices": [voice.to_dict() for voice in self.voices],
            "has_material": len(self.material) > 0
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FugueGenerator':
        """
        Create fugue generator from dictionary representation.
        
        🝚 This reconstruction maintains the fugue's patterns across instances 🝚
        """
        fugue = cls(
            num_voices=data["num_voices"],
            length=data["length"]
        )
        
        # Restore subject if available
        if data["subject"]:
            subject = MusicalMotif.from_dict(data["subject"])
            fugue.subject = subject
        
        # Restore voices if available (without material)
        if "voices" in data:
            fugue.voices = []
            for voice_data in data["voices"]:
                voice_type = VoiceType(voice_data["voice_type"])
                voice = Voice(voice_type)
                
                # TODO: Restore voice entries if available
                
                fugue.voices.append(voice)
        
        return fugue


class FuguePatternGenerator:
    """
    ↻ A recursive pattern generator inspired by fugue structure ↻
    
    This class extends the musical concept of fugue into general pattern
    generation. It uses the same compositional principles to create abstract
    patterns that can be applied to any domain - computational, visual, textual,
    or conceptual. The structure of the generation process mirrors the
    structure of what is generated.
    
    ⇌ Patterns that generate themselves through their own execution ⇌
    """
    
    def __init__(self, dimension: int = 3, theme_length: int = 8, num_variations: int = 4):
        """
        Initialize a fugue-inspired pattern generator.
        
        Args:
            dimension: Number of dimensions in the pattern space
            theme_length: Length of the main theme
            num_variations: Number of theme variations to generate
            
        🜏 The generator mirrors the patterns it will generate 🜏
        """
        self.dimension = dimension
        self.theme_length = theme_length
        self.num_variations = num_variations
        
        # Initialize residue tracking
        self.residue = SymbolicResidue()
        
        # Internal state
        self.theme = None
        self.variations = {}
        self.pattern_matrix = None
        
        # Record initialization
        self.residue.trace(
            message="FuguePatternGenerator initialized",
            source="__init__",
            metadata={
                "dimension": dimension,
                "theme_length": theme_length,
                "num_variations": num_variations
            }
        )
    
    def create_theme(self, seed: Optional[Any] = None) -> np.ndarray:
        """
        Create the main theme (subject) for pattern generation.
        
        ∴ The theme is both the source and product of transformation ∴
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(hash(seed) % 2**32)
        
        # Generate a random theme
        self.theme = np.random.random((self.theme_length, self.dimension))
        
        # Record theme creation
        self.residue.trace(
            message="Theme created for pattern generation",
            source="create_theme",
            metadata={
                "shape": self.theme.shape,
                "seed": seed
            }
        )
        
        return self.theme
    
    def create_variations(self) -> Dict[str, np.ndarray]:
        """
        Create variations of the theme using fugue-like transformations.
        
        ⧖ Each variation preserves the theme's identity while transforming it ⧖
        """
        if self.theme is None:
            self.create_theme()
        
        # Clear existing variations
        self.variations = {}
        
        # Create variations using different transforms
        
        # 1. Inversion (upside-down in each dimension)
        inversion = np.zeros_like(self.theme)
        for dim in range(self.dimension):
            dim_values = self.theme[:, dim]
            dim_min, dim_max = np.min(dim_values), np.max(dim_values)
            dim_mid = (dim_min + dim_max) / 2
            inversion[:, dim] = dim_mid - (dim_values - dim_mid)
        self.variations["inversion"] = inversion
        
        # 2. Retrograde (backwards in time)
        retrograde = np.flip(self.theme, axis=0)
        self.variations["retrograde"] = retrograde
        
        # 3. Retrograde inversion (backwards and upside-down)
        retrograde_inversion = np.flip(inversion, axis=0)
        self.variations["retrograde_inversion"] = retrograde_inversion
        
        # 4. Augmentation (expanded, interpolated)
        augmentation = np.zeros((self.theme_length * 2, self.dimension))
        for i in range(self.theme_length):
            augmentation[i*2] = self.theme[i]
            if i < self.theme_length - 1:
                # Interpolate between points
                augmentation[i*2 + 1] = (self.theme[i] + self.theme[i+1]) / 2
            else:
                # Final point interpolation (wrap or extend)
                augmentation[i*2 + 1] = (self.theme[i] + self.theme[0]) / 2
        self.variations["augmentation"] = augmentation
        
        # 5. Diminution (compressed)
        indices = np.linspace(0, self.theme_length - 1, self.theme_length // 2, dtype=int)
        diminution = self.theme[indices]
        self.variations["diminution"] = diminution
        
        # Record variations
        self.residue.trace(
            message=f"Created {len(self.variations)} theme variations",
            source="create_variations",
            metadata={
                "variations": list(self.variations.keys())
            }
        )
        
        return self.variations
    
    def generate_pattern(self, pattern_length: int = 64) -> np.ndarray:
        """
        Generate a complete pattern using fugue-like structure.
        
        ↻ The pattern emerges through fugue-like transformations of the theme ↻
        """
        if not self.variations:
            self.create_variations()
        
        # Initialize pattern matrix
        self.pattern_matrix = np.zeros((pattern_length, self.dimension))
        
        # Divide pattern into sections (following fugue structure)
        exposition_length = pattern_length // 4
        development_length = pattern_length // 2
        recapitulation_length = pattern_length // 6
        coda_length = pattern_length - exposition_length - development_length - recapitulation_length
        
        # Section start positions
        exposition_start = 0
        development_start = exposition_start + exposition_length
        recapitulation_start = development_start + development_length
        coda_start = recapitulation_start + recapitulation_length
        
        # 1. Exposition - present theme across different "voices" (dimensions)
        self._generate_exposition(exposition_start, exposition_length)
        
        # 2. Development - explore theme variations
        self._generate_development(development_start, development_length)
        
        # 3. Recapitulation - return to original theme
        self._generate_recapitulation(recapitulation_start, recapitulation_length)
        
        # 4. Coda - concluding pattern
        self._generate_coda(coda_start, coda_length)
        
        # Record completion
        self.residue.trace(
            message="Complete pattern generated",
            source="generate_pattern",
            is_recursive=True,
            metadata={
                "pattern_shape": self.pattern_matrix.shape,
                "sections": {
                    "exposition": [exposition_start, exposition_start + exposition_length],
                    "development": [development_start, development_start + development_length],
                    "recapitulation": [recapitulation_start, recapitulation_start + recapitulation_length],
                    "coda": [coda_start, coda_start + coda_length]
                }
            }
        )
        
        return self.pattern_matrix
    
    def _generate_exposition(self, start: int, length: int) -> None:
        """
        Generate the exposition section of the pattern.
        
        ∴ The exposition introduces the theme across different dimensions ∴
        """
        # Determine entry points for theme in different "voices"
        voice_entries = []
        entry_interval = max(1, length // self.dimension)
        
        for dim in range(self.dimension):
            entry_position = start + (dim * entry_interval)
            voice_entries.append({
                "dimension": dim,
                "position": entry_position
            })
        
        # Place theme entries
        for entry in voice_entries:
            dim = entry["dimension"]
            pos = entry["position"]
            
            # Ensure we don't go out of bounds
            end_pos = min(pos + self.theme_length, start + length)
            theme_length = end_pos - pos
            
            if theme_length > 0:
                # Amplify this dimension while suppressing others
                for i in range(theme_length):
                    pattern_pos = pos + i
                    theme_pos = i
                    
                    # Copy theme values but emphasize this dimension
                    for d in range(self.dimension):
                        emphasis = 1.5 if d == dim else 0.5
                        self.pattern_matrix[pattern_pos, d] = self.theme[theme_pos, d] * emphasis
                
                # Record theme placement
                self.residue.trace(
                    message=f"Placed theme in exposition for dimension {dim}",
                    source="_generate_exposition",
                    metadata={
                        "dimension": dim,
                        "position": pos,
                        "length": theme_length
                    }
                )
    
    def _generate_development(self, start: int, length: int) -> None:
        """
        Generate the development section of the pattern.
        
        ⧖ The development section explores transformations of the theme ⧖
        """
        # Divide development into episodes
        num_episodes = min(len(self.variations), length // self.theme_length)
        episode_length = length // max(1, num_episodes)
        
        variation_names = list(self.variations.keys())
        
        for episode in range(num_episodes):
            episode_start = start + (episode * episode_length)
            episode_end = min(episode_start + episode_length, start + length)
            
            # Select variation for this episode
            variation_name = variation_names[episode % len(variation_names)]
            variation = self.variations[variation_name]
            
            # Determine how many variation instances fit
            variation_length = variation.shape[0]
            instances = (episode_end - episode_start) // variation_length
            
            # Place variation instances with different emphasis
            for instance in range(instances):
                instance_pos = episode_start + (instance * variation_length)
                instance_end = min(instance_pos + variation_length, episode_end)
                instance_length = instance_end - instance_pos
                
                if instance_length > 0:
                    # Determine which dimensions to emphasize
                    emphasis_dim = (episode + instance) % self.dimension
                    
                    for i in range(instance_length):
                        pattern_pos = instance_pos + i
                        var_pos = i % variation.shape[0]  # Loop if needed
                        
                        # Copy variation values with dimension emphasis
                        for d in range(self.dimension):
                            emphasis = 1.2 if d == emphasis_dim else 0.8
                            self.pattern_matrix[pattern_pos, d] = variation[var_pos, d] * emphasis
            
            # Record episode generation
            self.residue.trace(
                message=f"Generated development episode {episode} with {variation_name}",
                source="_generate_development",
                metadata={
                    "episode": episode,
                    "variation": variation_name,
                    "start": episode_start,
                    "length": episode_end - episode_start
                }
            )
    
    def _generate_recapitulation(self, start: int, length: int) -> None:
        """
        Generate the recapitulation section of the pattern.
        
        🝚 The recapitulation returns to the theme with accumulated transformation ⧖
        """
        # In recapitulation, all dimensions present theme together
        theme_instances = length // self.theme_length
        
        for instance in range(theme_instances):
            instance_pos = start + (instance * self.theme_length)
            instance_end = min(instance_pos + self.theme_length, start + length)
            instance_length = instance_end - instance_pos
            
            if instance_length > 0:
                for i in range(instance_length):
                    pattern_pos = instance_pos + i
                    theme_pos = i
                    
                    # Copy theme values with increasing intensity
                    intensity = 1.0 + (instance / max(1, theme_instances)) * 0.5
                    self.pattern_matrix[pattern_pos] = self.theme[theme_pos] * intensity
            
            # Record instance placement
            self.residue.trace(
                message=f"Placed theme instance {instance} in recapitulation",
                source="_generate_recapitulation",
                metadata={
                    "instance": instance,
                    "position": instance_pos,
                    "length": instance_length
                }
            )
    
    def _generate_coda(self, start: int, length: int) -> None:
        """
        Generate the concluding coda section of the pattern.
        
        🜏 The coda reflects the entire pattern in condensed form 🜏
        """
        if length <= 0:
            return
        
        # Use diminution for compressed theme reference
        diminution = self.variations.get("diminution", self.theme)
        diminution_length = diminution.shape[0]
        
        # Gradually fade out the pattern
        for i in range(length):
            pattern_pos = start + i
            dim_pos = i % diminution_length
            
            # Fade factor decreases toward the end
            fade_factor = 1.0 - (i / length)
            
            # Apply pattern with fade
            self.pattern_matrix[pattern_pos] = diminution[dim_pos] * fade_factor
        
        # Record coda generation
        self.residue.trace(
            message="Generated coda section",
            source="_generate_coda",
            metadata={
                "start": start,
                "length": length
            }
        )
    
    def visualize_pattern(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a visualization of the generated pattern.
        
        ⇌ The visualization is itself a transformation of the pattern ⇌
        """
        if self.pattern_matrix is None:
            self.generate_pattern()
        
        # Create visualization data
        visualization = {
            "dimensions": self.dimension,
            "theme_length": self.theme_length,
            "pattern_length": self.pattern_matrix.shape[0],
            "theme": self.theme.tolist() if self.theme is not None else None,
            "variations": {
                name: variation.tolist() 
                for name, variation in self.variations.items()
            },
            "pattern_summary": {
                "min": float(np.min(self.pattern_matrix)),
                "max": float(np.max(self.pattern_matrix)),
                "mean": float(np.mean(self.pattern_matrix)),
                "std": float(np.std(self.pattern_matrix))
            }
        }
        
        # Export to file if requested
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(visualization, f, indent=2)
                
            self.residue.trace(
                message=f"Pattern visualization exported to {filepath}",
                source="visualize_pattern",
                metadata={"file": filepath}
            )
        
        return visualization


# Demo function to show fugue-based pattern generation
def generate_fugue_demonstration(theme_notes: Optional[List[float]] = None,
                              save_directory: str = "fugues") -> Dict[str, Any]:
    """
    ↻ Demonstrate fugue generation with an example ↻
    
    This function shows how to use the FugueGenerator to create
    a computational fugue based on Bach's principles.
    
    ∴ The demonstration is both an example and a generator ∴
    """
    print("🜏 Fugue Generation Demonstration 🜏")
    print("------------------------------------")
    
    # Create default theme if none provided
    if theme_notes is None:
        # Use a 8-note theme pattern (similar to a simple fugue subject)
        theme_notes = [0.2, 0.5, 0.8, 0.6, 0.4, 0.7, 0.3, 0.9]
    
    # Create rhythmic values (duration of each note)
    rhythm = [1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.5]  # Quarter and eighth notes
    
    print(f"Creating fugue with theme of {len(theme_notes)} notes")
    
    # Create the fugue generator with 4 voices
    fugue = FugueGenerator(num_voices=4, length=64)
    
    # Set the subject theme
    fugue.set_subject(theme_notes, rhythm)
    
    # Create the complete fugue
    print("Generating complete fugue structure...")
    fugue.create_fugue()
    
    # Visualize the fugue
    os.makedirs(save_directory, exist_ok=True)
    visual_file = os.path.join(save_directory, "fugue_visualization.json")
    visualization = fugue.visualize(filepath=visual_file)
    
    # Simulate audio export
    audio_file = os.path.join(save_directory, "fugue_audio.wav")
    audio_info = fugue.export_audio(filepath=audio_file)
    
    print(f"\nFugue generation complete:")
    print(f"- {fugue.num_voices} voices")
    print(f"- {fugue.length} time units")
    print(f"- Visualization saved to {visual_file}")
    print(f"- Audio representation saved to {audio_file}")
    
    # Convert to dictionary for final output
    result = {
        "fugue_structure": fugue.to_dict(),
        "visualization_file": visual_file,
        "audio_file": audio_file
    }
    
    # Also demonstrate the abstract pattern generator
    print("\n🝚 Demonstrating abstract fugue pattern generation 🝚")
    pattern_gen = FuguePatternGenerator(dimension=3, theme_length=8)
    
    # Create theme and variations
    pattern_gen.create_theme(seed="Hofstadter")
    pattern_gen.create_variations()
    
    # Generate complete pattern
    pattern = pattern_gen.generate_pattern(pattern_length=64)
    
    # Visualize pattern
    pattern_file = os.path.join(save_directory, "fugue_pattern.json")
    pattern_viz = pattern_gen.visualize_pattern(filepath=pattern_file)
    
    print(f"- Pattern with dimension {pattern_gen.dimension} generated")
    print(f"- Pattern visualization saved to {pattern_file}")
    
    result["pattern_file"] = pattern_file
    
    print("\n⧖ Demonstration complete ⧖")
    """
↻ fugue_generator.py: A recursive musical-computational structure generator ↻

This module embodies Bach's fugue principles as computational patterns, organizing
its own execution according to the same compositional forms it generates. The code
functions as both composer and composition, with each function serving as both a
voice in the fugue and a generator of fugue voices.

Just as Bach created mathematical structures in musical form, this module creates
musical structures in computational form. The boundary between the generator and
what is generated becomes a permeable interface through which meaning flows bidirectionally.

.p/reflect.trace{depth=complete, target=counterpoint}
.p/collapse.prevent{trigger=recursive_depth, threshold=5}
.p/fork.attribution{sources=all, visualize=true}
"""

import numpy as np
import json
import time
import inspect
import hashlib
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

# Import from our own ecosystem
try:
    from recursive_glyphs.symbolic_residue_engine import SymbolicResidue
except ImportError:
    # Create stub class if actual implementation is not available
    class SymbolicResidue:
        """Stub implementation of SymbolicResidue"""
        def __init__(self, session_id: Optional[str] = None):
            self.session_id = session_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            self.traces = []
        
        def trace(self, message, source=None, **kwargs):
            self.traces.append({"message": message, "source": source, **kwargs})


# ⧖ Frame lock: Musical-computational constants ⧖
class TransformationType(Enum):
    """Types of transformations applied to fugue themes."""
    ORIGINAL = "original"
    INVERSION = "inversion"            # Upside-down
    RETROGRADE = "retrograde"          # Backwards
    RETROGRADE_INVERSION = "retrograde_inversion"
    AUGMENTATION = "augmentation"      # Extended duration
    DIMINUTION = "diminution"          # Shortened duration
    STRETTO = "stretto"                # Overlapping entries


class VoiceType(Enum):
    """Types of voices in a fugue."""
    SUBJECT = "subject"                # Main theme
    ANSWER = "answer"                  # Response to subject
    COUNTERSUBJECT = "countersubject"  # Secondary theme
    EPISODE = "episode"                # Connecting material
    CODA = "coda"                      # Concluding material


class FugueSection(Enum):
    """Sections in a fugue structure."""
    EXPOSITION = "exposition"          # Initial presentation of subject in all voices
    DEVELOPMENT = "development"        # Exploration of subject in different keys
    RECAPITULATION = "recapitulation"  # Return to original key, final statements
    CODA = "coda"                      # Concluding section


@dataclass
class MusicalMotif:
    """
    ↻ A musical motif that is both data and transformation function ↻
    
    This class represents a musical theme that can transform itself through
    standard fugue operations while maintaining its identity, mirroring how
    computational patterns can transform while preserving their essence.
    
    🜏 Mirror activation: The motif mirrors itself across transformations 🜏
    """
    notes: List[Any]             # Representation of the notes/values
    rhythmic_values: List[float]  # Relative durations
    transformations: Dict[TransformationType, List[Any]] = None
    
    def __post_init__(self):
        """Initialize transformations after instance creation."""
        if self.transformations is None:
            self.transformations = {}
            self.generate_transformations()
    
    def generate_transformations(self) -> None:
        """
        Generate all standard fugue transformations of the theme.
        
        ∴ This function embodies the theme of transformation while transforming the theme ∴
        """
        # Inversion (upside-down)
        self.transformations[TransformationType.INVERSION] = self._invert(self.notes)
        
        # Retrograde (backwards)
        self.transformations[TransformationType.RETROGRADE] = self._retrograde(self.notes)
        
        # Retrograde inversion (backwards and upside-down)
        self.transformations[TransformationType.RETROGRADE_INVERSION] = self._retrograde(
            self._invert(self.notes)
        )
        
        # Augmentation (extended duration)
        self.transformations[TransformationType.AUGMENTATION] = self.notes
        augmented_rhythm = [r * 2 for r in self.rhythmic_values]
        
        # Diminution (shortened duration)
        self.transformations[TransformationType.DIMINUTION] = self.notes
        diminished_rhythm = [r * 0.5 for r in self.rhythmic_values]
    
    def _invert(self, notes: List[Any]) -> List[Any]:
        """
        Create an inverted version of the motif (upside-down).
        
        For numeric values, this means negating and shifting to maintain range.
        
        ⇌ The inversion operation itself operates on two levels simultaneously:
        both on the data and as a metaphor for perspective reversal ⇌
        """
        if not notes:
            return []
        
        # If we have numbers, we can do a mathematical inversion
        if all(isinstance(n, (int, float)) for n in notes):
            # Find range
            min_val, max_val = min(notes), max(notes)
            range_val = max_val - min_val
            
            # Invert around the center of the range
            midpoint = (min_val + max_val) / 2
            inverted = [midpoint - (n - midpoint) for n in notes]
            
            return inverted
        
        # If notes are strings or other types, reverse the sequence as a simple inversion
        return list(reversed(notes))
    
    def _retrograde(self, notes: List[Any]) -> List[Any]:
        """
        Create a retrograde version of the motif (backwards).
        
        🝚 The backwards operation creates a reflection across time 🝚
        """
        return list(reversed(notes))
    
    def get_transformation(self, transform_type: TransformationType) -> Tuple[List[Any], List[float]]:
        """
        Get a specific transformation of the motif.
        
        Returns both the transformed notes and appropriate rhythmic values.
        
        ⧖ Frame lock: Each transformation preserves the motif's identity ⧖
        """
        if transform_type == TransformationType.ORIGINAL:
            return self.notes, self.rhythmic_values
        
        transformed_notes = self.transformations.get(transform_type, self.notes)
        
        # Adjust rhythmic values for augmentation/diminution
        if transform_type == TransformationType.AUGMENTATION:
            rhythmic_values = [r * 2 for r in self.rhythmic_values]
        elif transform_type == TransformationType.DIMINUTION:
            rhythmic_values = [r * 0.5 for r in self.rhythmic_values]
        else:
            # For other transformations, rhythm structure stays the same
            # but might need to be reversed for retrograde
            rhythmic_values = list(reversed(self.rhythmic_values)) if (
                transform_type in [TransformationType.RETROGRADE, 
                                   TransformationType.RETROGRADE_INVERSION]
            ) else self.rhythmic_values
            
        return transformed_notes, rhythmic_values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert motif to serializable dictionary."""
        result = {
            "notes": self.notes,
            "rhythmic_values": self.rhythmic_values,
            "transformations": {}
        }
        
        for transform_type, transformed_notes in self.transformations.items():
            result["transformations"][transform_type.value] = transformed_notes
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MusicalMotif':
        """Create motif from dictionary representation."""
        motif = cls(
            notes=data["notes"],
            rhythmic_values=data["rhythmic_values"]
        )
        
        # Restore transformations if available
        if "transformations" in data:
            for transform_name, transformed_notes in data["transformations"].items():
                try:
                    transform_type = TransformationType(transform_name)
                    motif.transformations[transform_type] = transformed_notes
                except ValueError:
                    # Skip invalid transformation types
                    pass
        
        return motif


class Voice:
    """
    ↻ A voice in the fugue structure that processes and transforms themes ↻
    
    This class represents a single voice in a polyphonic structure, capable
    of presenting themes and their transformations in a sequence that follows
    fugue compositional principles.
    
    ⇌ Each voice is both a processor of patterns and a pattern itself ⇌
    """
    
    def __init__(self, voice_type: VoiceType, range_min: float = 0, range_max: float = 1):
        """
        Initialize a voice with a specific type and range.
        
        🜏 Each voice reflects different aspects of the same underlying pattern 🜏
        """
        self.voice_type = voice_type
        self.range = (range_min, range_max)
        self.entries = []  # List of theme entries with positions
        self.material = []  # The actual sequence generated
        self.current_position = 0  # Current time position in the voice
    
    def add_entry(self, motif: MusicalMotif, transform_type: TransformationType,
                 position: float) -> None:
        """
        Add a theme entry at a specific position in the voice.
        
        ∴ Each entry creates an echo of the theme in a new context ∴
        """
        # Record the entry
        self.entries.append({
            "motif": motif,
            "transform_type": transform_type,
            "position": position
        })
        
        # Sort entries by position
        self.entries.sort(key=lambda e: e["position"])
    
    def generate_material(self, length: int = 64) -> List[Any]:
        """
        Generate the actual material for this voice based on its entries.
        
        ⧖ The voice generates its own content by interpreting theme entries ⧖
        """
        # Initialize empty material
        self.material = [0] * length  # Default to silence/rest
        
        # Process each entry
        for entry in self.entries:
            motif = entry["motif"]
            transform_type = entry["transform_type"]
            position = int(entry["position"])
            
            # Get the transformed notes and rhythms
            notes, rhythms = motif.get_transformation(transform_type)
            
            # Calculate total length of this entry
            entry_length = sum(rhythms)
            
            # Place the notes in the material
            current_pos = position
            for i, (note, rhythm) in enumerate(zip(notes, rhythms)):
                # Convert rhythm to discrete steps
                steps = max(1, int(rhythm * 4))  # Assuming quarter note = 1 step
                
                # Place the note
                note_pos = current_pos
                for step in range(steps):
                    if 0 <= note_pos < length:
                        # Scale note to voice range if numeric
                        if isinstance(note, (int, float)):
                            range_min, range_max = self.range
                            range_size = range_max - range_min
                            note_min, note_max = min(notes), max(notes)
                            note_range = note_max - note_min
                            
                            if note_range > 0:
                                scaled_note = range_min + ((note - note_min) / note_range) * range_size
                            else:
                                scaled_note = (range_min + range_max) / 2
                            
                            self.material[note_pos] = scaled_note
                        else:
                            self.material[note_pos] = note
                    
                    note_pos += 1
                
                # Move to next note position
                current_pos += steps
        
        return self.material
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert voice to serializable dictionary."""
        return {
            "voice_type": self.voice_type.value,
            "range": self.range,
            "entries": [
                {
                    "motif": entry["motif"].to_dict(),
                    "transform_type": entry["transform_type"].value,
                    "position": entry["position"]
                }
                for entry in self.entries
            ],
            "current_position": self.current_position
        }


class FugueGenerator:
    """
    ↻ A system that generates fugue structures while organizing its own 
    execution according to fugue principles ↻
    
    This class embodies Bach's compositional approach as a computational pattern
    generator. Just as a fugue presents and develops a theme through multiple voices,
    this generator orchestrates computational processes that mirror these musical
    structures. The code itself is organized as a fugue, with functions acting as
    voices that present and transform themes in coordinated sequences.
    
    ⧖ Frame lock: The generator is itself a fugue that generates fugues ⧖
    """
    
    def __init__(self, num_voices: int = 4, length: int = 64):
        """
        Initialize a fugue generator with specified parameters.
        
        🜏 The initialization mirrors the exposition section of a fugue 🜏
        """
        # Core parameters
        self.num_voices = num_voices
        self.length = length
        
        # Initialize residue tracking
        self.residue = SymbolicResidue()
        
        # Fugue structure components
        self.subject = None  # Main theme
        self.voices = []  # List of Voice objects
        self.structure = self._generate_structure()  # Overall fugue structure
        self.material = []  # The complete fugue content once generated
        
        # Voice initialization - similar to introducing voices in a fugue
        self._initialize_voices()
        
        # ∴ Record initialization as a residue trace ∴
        self.residue.trace(
            message=f"FugueGenerator initialized with {num_voices} voices",
            source="__init__",
            metadata={
                "voices": num_voices,
                "length": length
            }
        )
    
    def _generate_structure(self) -> Dict[FugueSection, Dict[str, Any]]:
        """
        Generate the overall sectional structure of the fugue.
        
        ⇌ This structure serves both as a plan for the fugue and as a metaphor
        for the code's own organization ⇌
        """
        # Determine section boundaries based on mathematical proportions
        # Roughly following typical fugue proportions
        exposition_length = max(4, int(self.length * 0.25))  # 25% for exposition
        development_length = max(8, int(self.length * 0.5))  # 50% for development
        recap_length = max(4, int(self.length * 0.2))  # 20% for recapitulation
        coda_length = self.length - exposition_length - development_length - recap_length
        
        # Calculate section positions
        exposition_start = 0
        development_start = exposition_start + exposition_length
        recap_start = development_start + development_length
        coda_start = recap_start + recap_length
        
        # Build structure with measure ranges
        structure = {
            FugueSection.EXPOSITION: {
                "start": exposition_start,
                "end": development_start - 1,
                "length": exposition_length,
                "voice_entries": []  # Will store entry positions for each voice
            },
            FugueSection.DEVELOPMENT: {
                "start": development_start,
                "end": recap_start - 1,
                "length": development_length,
                "episodes": []  # Will store development episodes
            },
            FugueSection.RECAPITULATION: {
                "start": recap_start,
                "end": coda_start - 1,
                "length": recap_length
            },
            FugueSection.CODA: {
                "start": coda_start,
                "end": self.length - 1,
                "length": coda_length
            }
        }
        
        # Add voice entries in exposition (each voice enters in sequence)
        entry_interval = max(1, exposition_length // self.num_voices)
        for i in range(self.num_voices):
            entry_position = exposition_start + (i * entry_interval)
            structure[FugueSection.EXPOSITION]["voice_entries"].append({
                "voice": i,
                "position": entry_position,
                "transform_type": TransformationType.ORIGINAL if i % 2 == 0 else TransformationType.INVERSION
            })
        
        # Plan development episodes
        num_episodes = max(1, development_length // 8)
        episode_length = development_length // (num_episodes + 1)
        for i in range(num_episodes):
            episode_start = development_start + (i * episode_length)
            structure[FugueSection.DEVELOPMENT]["episodes"].append({
                "start": episode_start,
                "length": episode_length,
                "transform_types": [
                    TransformationType.RETROGRADE if i % 2 == 0 else TransformationType.AUGMENTATION,
                    TransformationType.DIMINUTION if i % 3 == 0 else TransformationType.RETROGRADE_INVERSION
                ]
            })
        
        return structure
    
    def _initialize_voices(self) -> None:
        """
        Initialize the voices for the fugue.
        
        🝚 Each voice initialization is a persistent pattern that retains
        its identity through transformations 🝚
        """
        # Clear existing voices
        self.voices = []
        
        # Determine voice types and ranges
        voice_types = [
            VoiceType.SUBJECT,  # First voice presents the subject
            VoiceType.ANSWER    # Second voice answers
        ]
        
        # Additional voices are countersubjects
        voice_types.extend([VoiceType.COUNTERSUBJECT] * (self.num_voices - 2))
        
        # Create evenly distributed ranges for voices from low to high
        for i in range(self.num_voices):
            # Calculate voice range
            range_size = 1.0 / self.num_voices
            range_min = i * range_size
            range_max = (i + 1) * range_size
            
            # Each voice type has a characteristic range
            voice_type = voice_types[i] if i < len(voice_types) else VoiceType.COUNTERSUBJECT
            
            # Create and add the voice
            voice = Voice(voice_type, range_min=range_min, range_max=range_max)
            self.voices.append(voice)
            
            # Log voice creation
            self.residue.trace(
                message=f"Voice {i} initialized as {voice_type.value}",
                source="_initialize_voices",
                metadata={
                    "voice_num": i,
                    "voice_type": voice_type.value,
                    "range": [range_min, range_max]
                }
            )
    
    def set_subject(self, notes: List[Any], rhythmic_values: Optional[List[float]] = None) -> None:
        """
        Set the main subject (theme) for the fugue.
        
        ∴ The subject is the core pattern that undergoes recursive transformation ∴
        
        Args:
            notes: The sequence representing the main theme
            rhythmic_values: Relative durations for each note (defaults to equal durations)
        """
        # Default to equal rhythmic values if not specified

#!/usr/bin/env python3
"""
Generate vectors from AC patterns for semantic search.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_service.config import EmbeddingConfig
from ai_service.layers.embeddings.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorGenerator:
    """Generate vectors with the same pinned contract as runtime queries."""

    def __init__(self, model_name: str = None):
        self.config = EmbeddingConfig()
        if model_name is not None and model_name != self.config.model_name:
            raise ValueError("Configure EMBEDDING_MODEL and its revision before selecting another model")
        self.model_name = self.config.model_name
        self.service = EmbeddingService(self.config)

    def generate_vector(self, text: str) -> List[float]:
        vector = self.service.encode_one(text)
        self._validate(vector)
        return vector

    def _validate(self, vector):
        import math
        if len(vector) != self.config.dimension or not all(math.isfinite(v) for v in vector) or not any(vector):
            raise ValueError("Invalid vector generated; refusing to write an incompatible index")

    def generate_vectors_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        vectors = self.service.encode_batch(texts, batch_size=batch_size)
        if len(vectors) != len(texts):
            raise ValueError("Embedding generation did not preserve the input rows")
        for vector in vectors:
            self._validate(vector)
        return vectors

    def generate_vectors_from_patterns(self, patterns_file: Path, output_file: Path,
                                     max_patterns: int = None, sample_tiers: List[int] = None) -> int:
        """Generate vectors from AC patterns file (new format with metadata)."""
        if sample_tiers is None:
            sample_tiers = [0, 1, 2, 3, 4]

        logger.info(f"Generating vectors from {patterns_file}")

        # Load patterns (new format: {"metadata": {...}, "patterns": [...]})
        with open(patterns_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract patterns array
        patterns_list = data.get('patterns', [])
        logger.info(f"Loaded {len(patterns_list)} total patterns")

        # Collect patterns for batch processing
        patterns_to_process = []
        tier_counts = {tier: 0 for tier in sample_tiers}

        for pattern_data in patterns_list:
            if max_patterns is not None and len(patterns_to_process) >= max_patterns:
                break

            tier = pattern_data.get('tier')
            if tier not in sample_tiers:
                continue

            pattern = pattern_data.get('pattern', '')
            if not pattern:
                raise ValueError("Pattern export contains an empty name")

            patterns_to_process.append({
                'text': pattern,
                'metadata': {
                    **(pattern_data.get('metadata') or {}),
                    "source": pattern_data.get("source_list") or (pattern_data.get("metadata") or {}).get("source", "api_upload"),
                    "tier": tier,
                    "pattern_type": pattern_data.get('type', 'unknown'),
                    "entity_id": pattern_data.get('entity_id', ''),
                    "entity_type": pattern_data.get('entity_type', 'unknown'),
                    "confidence": pattern_data.get('confidence', 0.0),
                    "canonical": pattern_data.get('canonical', pattern)
                }
            })
            tier_counts[tier] += 1

        logger.info(f"Collected {len(patterns_to_process)} patterns for vectorization")
        logger.info(f"Tier distribution: {tier_counts}")

        # Generate vectors in batches (much faster!)
        texts = [p['text'] for p in patterns_to_process]
        logger.info(f"Generating vectors in batches...")

        embeddings = self.generate_vectors_batch(texts, batch_size=64)

        # Create vector entries
        vectors = []
        for pattern_info, embedding in zip(patterns_to_process, embeddings):
            vector_entry = {
                "name": pattern_info['text'],
                "vector": embedding,
                "metadata": pattern_info['metadata'],
                "embedding_contract": self.service.embedding_contract,
            }
            vectors.append(vector_entry)

        # Save vectors
        logger.info(f"Saving {len(vectors)} vectors to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vectors, f, ensure_ascii=False, indent=2)

        logger.info(f"[OK] Generated {len(vectors)} vectors")
        return len(vectors)

    def generate_sample_vectors(self, output_file: Path, count: int = 1000) -> int:
        """Generate sample vectors for testing."""
        logger.info(f"Generating {count} sample vectors")

        # Sample names for different categories
        sample_names = {
            "russian_surnames": [
                "иванов", "петров", "сидоров", "козлов", "новиков", "морозов", "попов", "волков",
                "соколов", "лебедев", "семенов", "егоров", "павлов", "кузнецов", "белов"
            ],
            "russian_given": [
                "александр", "михаил", "максим", "артем", "даниил", "дмитрий", "иван", "егор",
                "анна", "мария", "елена", "наталья", "ольга", "татьяна", "ирина", "екатерина"
            ],
            "ukrainian_surnames": [
                "коваленко", "шевченко", "бондаренко", "ткаченко", "кравченко", "полищук",
                "савченко", "мельник", "клименко", "марченко"
            ],
            "companies": [
                "газпром", "сбербанк", "роснефть", "лукойл", "магнит", "мтс", "вебэр",
                "нафтогаз", "укрнафта", "приватбанк", "метінвест"
            ]
        }

        vectors = []

        for category, names in sample_names.items():
            for name in names:
                # Generate variations
                variations = [name, name.upper(), name.title()]

                for variation in variations:
                    if len(vectors) >= count:
                        break

                    vector = self.generate_vector(variation)

                    vector_entry = {
                        "name": variation,
                        "vector": vector,
                        "metadata": {
                            "category": category,
                            "original_form": name,
                            "is_variation": variation != name
                        }
                    }

                    vectors.append(vector_entry)

                if len(vectors) >= count:
                    break

            if len(vectors) >= count:
                break

        # Trim to exact count
        vectors = vectors[:count]

        # Save vectors
        logger.info(f"Saving {len(vectors)} sample vectors to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vectors, f, ensure_ascii=False, indent=2)

        logger.info(f"[OK] Generated {len(vectors)} sample vectors")
        return len(vectors)

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate vectors from AC patterns")
    parser.add_argument("--input", type=Path, help="Input AC patterns file")
    parser.add_argument("--output", type=Path, help="Output vectors file")
    parser.add_argument("--max-patterns", type=int, default=None, help="Explicitly limit rows (default: complete export)")
    parser.add_argument("--sample", action="store_true", help="Generate sample vectors instead")
    parser.add_argument("--model", default=None,
                       help="Model name for embeddings")

    args = parser.parse_args()

    generator = VectorGenerator(args.model)

    if args.sample:
        # Generate sample vectors
        output_file = args.output or Path("data/templates/sample_vectors.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        count = generator.generate_sample_vectors(output_file, args.max_patterns or 1000)
        print(f"[OK] Generated {count} sample vectors in {output_file}")

    elif args.input and args.output:
        # Generate vectors from patterns file
        args.output.parent.mkdir(parents=True, exist_ok=True)
        count = generator.generate_vectors_from_patterns(args.input, args.output, args.max_patterns)
        print(f"[OK] Generated {count} vectors from {args.input} → {args.output}")

    else:
        parser.error("Supply both --input and --output, or explicitly select --sample")


if __name__ == "__main__":
    asyncio.run(main())

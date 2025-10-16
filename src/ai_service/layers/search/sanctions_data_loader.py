"""
Sanctions data loader for fuzzy search.
Loads and caches sanctions lists from various sources for fuzzy matching.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import hashlib
import os
from dataclasses import field

from ...data.resources import PACKAGE_DATA_DIR

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from ...utils.logging_config import get_logger
from ...utils.profiling import profile_function


@dataclass
class SanctionEntry:
    """Single sanctions list entry."""
    name: str
    entity_type: str  # "person", "organization"
    source: str  # "ofac", "eu", "uk", etc.
    list_name: str  # "SDN", "EU Sanctions", etc.
    aliases: List[str] = None
    birth_date: Optional[str] = None
    nationality: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class SanctionsDataset:
    """Complete sanctions dataset for fuzzy matching."""
    persons: List[SanctionEntry]
    organizations: List[SanctionEntry]
    all_names: List[str]  # All names + aliases for fuzzy search
    name_to_entry: Dict[str, SanctionEntry]  # Quick lookup
    loaded_at: datetime
    sources: List[str]
    total_entries: int
    source_manifest: Dict[str, str] = field(default_factory=dict)
    name_to_entries: Dict[str, List[SanctionEntry]] = field(default_factory=dict)

    def __post_init__(self):
        # A shared name or alias must not discard another sanctioned entity.
        self.name_to_entries = {}
        for entry in self.persons + self.organizations:
            for name in dict.fromkeys([entry.name, *entry.aliases]):
                self.name_to_entries.setdefault(name, []).append(entry)
        self.all_names = sorted(self.name_to_entries)
        self.name_to_entry = {name: entries[0] for name, entries in self.name_to_entries.items()}

    def get_person_names(self) -> List[str]:
        """Get all person names including aliases."""
        names = []
        for entry in self.persons:
            names.append(entry.name)
            names.extend(entry.aliases)
        return names

    def get_org_names(self) -> List[str]:
        """Get all organization names including aliases."""
        names = []
        for entry in self.organizations:
            names.append(entry.name)
            names.extend(entry.aliases)
        return names


class SanctionsDataLoader:
    """Loads and manages sanctions data for fuzzy search."""

    def __init__(self, data_dir: Optional[Path] = None, cache_ttl_hours: int = 24,
                 *, allow_demo: bool = False, cache_dir: Optional[Path] = None):
        self.logger = get_logger(__name__)

        if data_dir is None:
            data_dir = os.environ.get("SANCTIONS_DATA_DIR") or PACKAGE_DATA_DIR
        self.data_dir = Path(data_dir).resolve()
        self.allow_demo = allow_demo
        self._source_state = None
        self._source_manifest = {}

        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._cached_dataset: Optional[SanctionsDataset] = None
        cache_dir = Path(cache_dir) if cache_dir is not None else Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        ) / "ai_service" / "sanctions"
        source_key = hashlib.sha256(str(self.data_dir).encode()).hexdigest()[:24]
        self._cache_file = cache_dir / f"{source_key}-{'demo' if allow_demo else 'real'}.json"

        self.logger.info(f"SanctionsDataLoader initialized with data_dir: {self.data_dir}")

        # Log available files
        if self.data_dir.exists():
            files = list(self.data_dir.glob("*.json"))
            self.logger.info(f"Available sanctions files: {[f.name for f in files]}")
        else:
            self.logger.warning(f"Sanctions data directory does not exist: {self.data_dir}")

    @profile_function("sanctions.load_dataset")
    async def load_dataset(self, force_reload: bool = False) -> SanctionsDataset:
        """
        Load complete sanctions dataset.

        Args:
            force_reload: Force reload from source files

        Returns:
            Complete sanctions dataset
        """
        # Stat on every access; hash only when a source changes. Old cache files
        # without a matching manifest can never replace current source data.
        manifest = await asyncio.to_thread(self._get_source_manifest)
        if not force_reload and self._cached_dataset:
            if (datetime.now() - self._cached_dataset.loaded_at < self.cache_ttl
                    and self._cached_dataset.source_manifest == manifest):
                self.logger.debug("Using cached sanctions dataset")
                return self._cached_dataset

        # Try to load from cache file
        if not force_reload and await self._load_from_cache():
            self.logger.info(f"[OK] Loaded from cache: {self._cached_dataset.total_entries} entries")
            return self._cached_dataset

        # Load from source files
        self.logger.info("Loading sanctions data from source files (cache miss or expired)...")
        dataset = await self._load_from_sources()

        # Cache the dataset
        await self._save_to_cache(dataset)
        self._cached_dataset = dataset

        self.logger.info(f"Loaded {dataset.total_entries} sanctions entries from {len(dataset.sources)} sources")
        return dataset

    def _get_source_manifest(self) -> Dict[str, str]:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Sanctions data directory does not exist: {self.data_dir}")
        names = {"sanctioned_persons.json", "sanctioned_companies.json",
                 "terrorism_black_list.json", "ofac_sdn.json", "eu_sanctions.json",
                 "uk_sanctions.json"}
        files = sorted(p for p in self.data_dir.glob("*.json")
                       if p.name in names or p.name.startswith("custom_"))
        state = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns,
                       p.stat().st_ctime_ns) for p in files)
        if state != self._source_state:
            self._source_manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            self._source_state = state
        return dict(self._source_manifest)

    async def _load_from_cache(self) -> bool:
        """Try to load dataset from cache file."""
        try:
            if not self._cache_file.exists():
                return False

            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self._cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.loads(await f.read())
            else:
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

            if (cache_data.get('schema_version') != 2
                    or cache_data.get('source_manifest') != self._source_manifest
                    or cache_data.get('allow_demo') != self.allow_demo):
                return False
            # Check cache age
            loaded_at = datetime.fromisoformat(cache_data['loaded_at'])
            if datetime.now() - loaded_at > self.cache_ttl:
                self.logger.debug("Cache expired")
                return False

            # Reconstruct dataset
            persons = [SanctionEntry(**entry) for entry in cache_data['persons']]
            organizations = [SanctionEntry(**entry) for entry in cache_data['organizations']]

            self._cached_dataset = SanctionsDataset(
                persons=persons,
                organizations=organizations,
                all_names=cache_data['all_names'],
                name_to_entry=cache_data['name_to_entry'],
                loaded_at=loaded_at,
                sources=cache_data['sources'],
                total_entries=cache_data['total_entries'],
                source_manifest=cache_data['source_manifest'],
            )

            # Rebuild name_to_entry mapping with actual objects
            name_to_entry = {}
            all_entries = persons + organizations
            for entry in all_entries:
                name_to_entry[entry.name] = entry
                for alias in entry.aliases:
                    name_to_entry[alias] = entry

            self.logger.debug("Successfully loaded from cache")
            return True

        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return False

    async def _save_to_cache(self, dataset: SanctionsDataset):
        """Save dataset to cache file."""
        try:
            # Convert to JSON-serializable format
            cache_data = {
                'schema_version': 2,
                'source_manifest': dataset.source_manifest,
                'allow_demo': self.allow_demo,
                'persons': [self._entry_to_dict(entry) for entry in dataset.persons],
                'organizations': [self._entry_to_dict(entry) for entry in dataset.organizations],
                'all_names': dataset.all_names,
                'name_to_entry': {name: self._entry_to_dict(entry) for name, entry in dataset.name_to_entry.items()},
                'loaded_at': dataset.loaded_at.isoformat(),
                'sources': dataset.sources,
                'total_entries': dataset.total_entries
            }

            # Atomic replacement prevents readers from seeing partial JSON.
            def write_cache():
                import tempfile
                self._cache_file.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                        dir=self._cache_file.parent, delete=False) as f:
                    temporary = Path(f.name)
                    try:
                        json.dump(cache_data, f, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                        os.replace(temporary, self._cache_file)
                    finally:
                        temporary.unlink(missing_ok=True)
            await asyncio.to_thread(write_cache)

            self.logger.debug("Successfully saved to cache")

        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")

    def _entry_to_dict(self, entry: SanctionEntry) -> Dict[str, Any]:
        """Convert SanctionEntry to dictionary."""
        return {
            'name': entry.name,
            'entity_type': entry.entity_type,
            'source': entry.source,
            'list_name': entry.list_name,
            'aliases': entry.aliases,
            'birth_date': entry.birth_date,
            'nationality': entry.nationality,
            'metadata': entry.metadata
        }

    async def _load_from_sources(self) -> SanctionsDataset:
        """Load sanctions data from various sources."""
        all_entries = []
        sources = []

        # Load from different file formats and sources
        loaders = [
            self._load_sanctioned_persons,   # Ukrainian sanctions persons
            self._load_sanctioned_companies, # Ukrainian sanctions companies
            self._load_terrorism_blacklist,  # Terrorism blacklist
            self._load_ofac_sdn,             # OFAC SDN list
            self._load_eu_sanctions,         # EU sanctions
            self._load_uk_sanctions,         # UK sanctions
            self._load_custom_lists,         # Custom sanctions lists
        ]
        if self.allow_demo:
            loaders.append(self._load_sample_data)

        for loader in loaders:
            try:
                entries, source_name = await loader()
                if entries:
                    all_entries.extend(entries)
                    sources.append(source_name)
                    self.logger.info(f"Loaded {len(entries)} entries from {source_name}")
                else:
                    self.logger.debug(f"No entries from {loader.__name__}")
            except Exception as e:
                raise ValueError(f"Failed to load sanctions source {loader.__name__}") from e

        # Split by type
        persons = [e for e in all_entries if e.entity_type == "person"]
        organizations = [e for e in all_entries if e.entity_type == "organization"]

        # Build name indexes
        all_names = []
        name_to_entry = {}

        for entry in all_entries:
            all_names.append(entry.name)
            name_to_entry[entry.name] = entry

            for alias in entry.aliases:
                all_names.append(alias)
                name_to_entry[alias] = entry

        dataset = SanctionsDataset(
            persons=persons,
            organizations=organizations,
            all_names=sorted(set(all_names)),
            name_to_entry=name_to_entry,
            loaded_at=datetime.now(),
            sources=sources,
            total_entries=len(all_entries),
            source_manifest=dict(self._source_manifest),
        )

        return dataset

    async def _load_sample_data(self) -> Tuple[List[SanctionEntry], str]:
        """Load sample sanctions data for testing."""
        entries = [
            # Ukrainian/Russian politicians and oligarchs
            SanctionEntry(
                name="Петро Порошенко",
                entity_type="person",
                source="sample",
                list_name="Sample List",
                aliases=["Petro Poroshenko", "Порошенко Петро Олексійович"],
                nationality="Ukraine"
            ),
            SanctionEntry(
                name="Владимир Путин",
                entity_type="person",
                source="sample",
                list_name="Sample List",
                aliases=["Vladimir Putin", "Путин Владимир Владимирович"],
                birth_date="1952-10-07",
                nationality="Russia"
            ),
            SanctionEntry(
                name="Ігор Коломойський",
                entity_type="person",
                source="sample",
                list_name="Sample List",
                aliases=["Igor Kolomoisky", "Коломойський Ігор Валерійович"],
                nationality="Ukraine"
            ),
            SanctionEntry(
                name="Рінат Ахметов",
                entity_type="person",
                source="sample",
                list_name="Sample List",
                aliases=["Rinat Akhmetov", "Ахметов Рінат Леонідович"],
                nationality="Ukraine"
            ),
            SanctionEntry(
                name="Алексей Навальный",
                entity_type="person",
                source="sample",
                list_name="Sample List",
                aliases=["Alexei Navalny", "Навальный Алексей Анатольевич"],
                birth_date="1976-06-04",
                nationality="Russia"
            ),
            # Test entry for fuzzy search
            SanctionEntry(
                name="Ковриков Роман Валерійович",
                entity_type="person",
                source="sample",
                list_name="Sample List",
                aliases=["Kovrykov Roman", "Роман Ковриков"],
                nationality="Ukraine"
            ),

            # Organizations
            SanctionEntry(
                name="Газпром",
                entity_type="organization",
                source="sample",
                list_name="Sample List",
                aliases=["Gazprom", "ПАО Газпром", "Gazprom PAO"]
            ),
            SanctionEntry(
                name="Роснефть",
                entity_type="organization",
                source="sample",
                list_name="Sample List",
                aliases=["Rosneft", "НК Роснефть", "Rosneft Oil Company"]
            ),
            SanctionEntry(
                name="Приватбанк",
                entity_type="organization",
                source="sample",
                list_name="Sample List",
                aliases=["PrivatBank", "АТ КБ ПриватБанк"]
            ),
        ]

        return entries, "Sample Data"

    async def _load_sanctioned_persons(self) -> Tuple[List[SanctionEntry], str]:
        """Load Ukrainian sanctioned persons data."""
        entries = []
        persons_file = self.data_dir / "sanctioned_persons.json"

        if not persons_file.exists():
            self.logger.debug("sanctioned_persons.json not found, skipping")
            return entries, "Ukrainian Persons"

        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(persons_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
            else:
                with open(persons_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            for item in data:
                # Extract aliases from different name fields
                aliases = list(item.get('aliases') or [])
                if item.get('name_ru') and item['name_ru'] != item['name']:
                    aliases.append(item['name_ru'])
                if item.get('name_en') and item['name_en'] != item['name']:
                    aliases.append(item['name_en'])

                # Create entry
                entry = SanctionEntry(
                    name=item.get('name', '').strip(),
                    entity_type="person",
                    source="ukrainian_sanctions",
                    list_name="Ukrainian Sanctioned Persons",
                    aliases=sorted({a.strip() for a in aliases if isinstance(a, str) and a.strip()}),
                    birth_date=item.get('birthdate'),
                    nationality=None,  # Not provided in this format
                    metadata={
                        'source_id': item.get('id', item.get('person_id')),
                        'person_id': item.get('person_id'),
                        'itn': item.get('itn'),
                        'itn_import': item.get('itn_import'),
                        'status': item.get('status')
                    }
                )

                # Only add if we have a valid name
                if entry.name:
                    entries.append(entry)

            self.logger.info(f"Loaded {len(entries)} Ukrainian sanctioned persons")

        except Exception as e:
            raise ValueError(f"Invalid sanctions source: {persons_file.name}") from e

        return entries, "Ukrainian Persons"

    async def _load_sanctioned_companies(self) -> Tuple[List[SanctionEntry], str]:
        """Load Ukrainian sanctioned companies data."""
        entries = []
        companies_file = self.data_dir / "sanctioned_companies.json"

        if not companies_file.exists():
            self.logger.debug("sanctioned_companies.json not found, skipping")
            return entries, "Ukrainian Companies"

        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(companies_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
            else:
                with open(companies_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            for item in data:
                # Create entry
                entry = SanctionEntry(
                    name=item.get('name', '').strip(),
                    entity_type="organization",
                    source="ukrainian_sanctions",
                    list_name="Ukrainian Sanctioned Companies",
                    aliases=sorted({a.strip() for a in [*(item.get('aliases') or []),
                                   item.get('name_en'), item.get('name_ru')]
                                   if isinstance(a, str) and a.strip()}),
                    birth_date=None,  # Not applicable for companies
                    nationality=None,  # Could extract from address
                    metadata={
                        'source_id': item.get('id', item.get('person_id')),
                        'person_id': item.get('person_id'),
                        'tax_number': item.get('tax_number'),
                        'reg_number': item.get('reg_number'),
                        'address': item.get('address'),
                        'status': item.get('status')
                    }
                )

                # Only add if we have a valid name
                if entry.name:
                    entries.append(entry)

            self.logger.info(f"Loaded {len(entries)} Ukrainian sanctioned companies")

        except Exception as e:
            raise ValueError(f"Invalid sanctions source: {companies_file.name}") from e

        return entries, "Ukrainian Companies"

    async def _load_terrorism_blacklist(self) -> Tuple[List[SanctionEntry], str]:
        """Load terrorism blacklist data."""
        entries = []
        blacklist_file = self.data_dir / "terrorism_black_list.json"

        if not blacklist_file.exists():
            self.logger.debug("terrorism_black_list.json not found, skipping")
            return entries, "Terrorism Blacklist"

        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(blacklist_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
            else:
                with open(blacklist_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            for item in data:
                # Extract name from aka_name
                name = item.get('aka_name', '').strip()
                if not name:
                    continue

                # Create entry
                entry = SanctionEntry(
                    name=name,
                    entity_type="person",  # Terrorism blacklist is typically persons
                    source="terrorism_blacklist",
                    list_name="Terrorism Blacklist",
                    metadata={
                        'date_entry': item.get('date_entry'),
                        'type_aka': item.get('type_aka'),
                        'quality_aka': item.get('quality_aka'),
                        'number_entry': item.get('number_entry')
                    }
                )

                entries.append(entry)

            self.logger.info(f"Loaded {len(entries)} terrorism blacklist entries")

        except Exception as e:
            raise ValueError(f"Invalid sanctions source: {blacklist_file.name}") from e

        return entries, "Terrorism Blacklist"

    async def _load_ofac_sdn(self) -> Tuple[List[SanctionEntry], str]:
        """Load OFAC SDN (Specially Designated Nationals) list."""
        entries = []
        sdn_file = self.data_dir / "ofac_sdn.json"

        if not sdn_file.exists():
            self.logger.debug("OFAC SDN file not found, skipping")
            return entries, "OFAC SDN"

        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(sdn_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
            else:
                with open(sdn_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            for item in data.get('entries', []):
                entry = SanctionEntry(
                    name=item.get('name', ''),
                    entity_type=item.get('type', 'person').lower(),
                    source="ofac",
                    list_name="SDN",
                    aliases=item.get('aliases', []),
                    birth_date=item.get('birth_date'),
                    nationality=item.get('nationality'),
                    metadata=item.get('metadata')
                )
                entries.append(entry)

        except Exception as e:
            raise ValueError(f"Invalid sanctions source: {sdn_file.name}") from e

        return entries, "OFAC SDN"

    async def _load_eu_sanctions(self) -> Tuple[List[SanctionEntry], str]:
        """Load EU sanctions list."""
        entries = []
        eu_file = self.data_dir / "eu_sanctions.json"

        if not eu_file.exists():
            self.logger.debug("EU sanctions file not found, skipping")
            return entries, "EU Sanctions"

        raise ValueError("EU source format is unsupported; convert to the documented custom entries schema")

    async def _load_uk_sanctions(self) -> Tuple[List[SanctionEntry], str]:
        """Load UK sanctions list."""
        entries = []
        uk_file = self.data_dir / "uk_sanctions.json"

        if not uk_file.exists():
            self.logger.debug("UK sanctions file not found, skipping")
            return entries, "UK Sanctions"

        raise ValueError("UK source format is unsupported; convert to the documented custom entries schema")

    async def _load_custom_lists(self) -> Tuple[List[SanctionEntry], str]:
        """Load custom sanctions lists."""
        entries = []

        # Look for custom JSON files in data directory
        custom_files = sorted(self.data_dir.glob("custom_*.json"))

        for custom_file in custom_files:
            try:
                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(custom_file, 'r', encoding='utf-8') as f:
                        data = json.loads(await f.read())
                else:
                    with open(custom_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                for item in data.get('entries', []):
                    entry = SanctionEntry(
                        name=item.get('name', ''),
                        entity_type=item.get('type', 'person').lower(),
                        source="custom",
                        list_name=custom_file.stem,
                        aliases=item.get('aliases', []),
                        birth_date=item.get('birth_date'),
                        nationality=item.get('nationality'),
                        metadata=item.get('metadata')
                    )
                    entries.append(entry)

                self.logger.debug(f"Loaded {len(data.get('entries', []))} entries from {custom_file}")

            except Exception as e:
                raise ValueError(f"Invalid sanctions source: {custom_file.name}") from e

        return entries, "Custom Lists"

    async def get_fuzzy_candidates(self, entity_type: Optional[str] = None) -> List[str]:
        """
        Get all names for fuzzy matching.

        Args:
            entity_type: Filter by 'person', 'organization', or None for all

        Returns:
            List of names for fuzzy matching
        """
        dataset = await self.load_dataset()

        if entity_type == "person":
            return dataset.get_person_names()
        elif entity_type == "organization":
            return dataset.get_org_names()
        else:
            return dataset.all_names

    async def find_entry(self, name: str) -> Optional[SanctionEntry]:
        """Find sanctions entry by exact name match."""
        dataset = await self.load_dataset()
        return dataset.name_to_entry.get(name)

    async def find_entries(self, name: str) -> List[SanctionEntry]:
        """Return all entities sharing an exact name or alias."""
        dataset = await self.load_dataset()
        return list(dataset.name_to_entries.get(name, []))

    async def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        if not self._cached_dataset:
            dataset = await self.load_dataset()
        else:
            dataset = self._cached_dataset

        return {
            'total_entries': dataset.total_entries,
            'persons': len(dataset.persons),
            'organizations': len(dataset.organizations),
            'unique_names': len(dataset.all_names),
            'sources': dataset.sources,
            'source_manifest': dataset.source_manifest,
            'demo_enabled': self.allow_demo,
            'loaded_at': dataset.loaded_at.isoformat(),
            'cache_age_hours': (datetime.now() - dataset.loaded_at).total_seconds() / 3600,
            'data_dir': str(self.data_dir)
        }

    async def clear_cache(self):
        """Clear cached data and force reload."""
        self._cached_dataset = None
        if self._cache_file.exists():
            self._cache_file.unlink()
        self.logger.info("Cache cleared")

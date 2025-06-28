"""
Museum API Data Collector for Ragamala Paintings.

This module provides functionality to scrape and collect Ragamala paintings
from various museum APIs and digital collections for SDXL fine-tuning.
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import logging
from datetime import datetime
import hashlib
import csv
from PIL import Image
import io

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.aws_utils import S3Manager

logger = setup_logger(__name__)

@dataclass
class MuseumConfig:
    """Configuration for museum API access."""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: float = 1.0  # seconds between requests
    max_retries: int = 3
    timeout: int = 30
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {
                'User-Agent': 'Ragamala-Research-Bot/1.0 (Educational Purpose)',
                'Accept': 'application/json'
            }
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'

@dataclass
class ArtworkMetadata:
    """Metadata structure for collected artworks."""
    id: str
    title: str
    artist: Optional[str]
    date: Optional[str]
    period: Optional[str]
    style: Optional[str]
    raga: Optional[str]
    region: Optional[str]
    medium: Optional[str]
    dimensions: Optional[str]
    description: Optional[str]
    keywords: List[str]
    museum: str
    collection: str
    image_url: str
    thumbnail_url: Optional[str]
    rights: Optional[str]
    source_url: str
    collected_date: str
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class MuseumAPICollector:
    """Base class for museum API collectors."""
    
    def __init__(self, config: MuseumConfig, output_dir: str = "data/raw"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.collected_items = []
        self.failed_items = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers=self.config.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                await asyncio.sleep(self.config.rate_limit)
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return None
    
    def extract_ragamala_keywords(self, text: str) -> List[str]:
        """Extract Ragamala-related keywords from text."""
        ragamala_terms = [
            'ragamala', 'raga', 'ragini', 'putra', 'bharata',
            'bhairav', 'malkauns', 'yaman', 'darbari', 'bageshri',
            'todi', 'puriya', 'marwa', 'bhimpalasi', 'kafi',
            'miniature', 'rajput', 'pahari', 'deccan', 'mughal',
            'kangra', 'basohli', 'mewar', 'bundi', 'golconda'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for term in ragamala_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        return found_keywords
    
    def is_ragamala_related(self, metadata: Dict) -> bool:
        """Check if artwork is Ragamala-related."""
        text_fields = [
            metadata.get('title', ''),
            metadata.get('description', ''),
            metadata.get('keywords', ''),
            metadata.get('subject', ''),
            metadata.get('culture', ''),
            metadata.get('classification', '')
        ]
        
        combined_text = ' '.join(str(field) for field in text_fields)
        keywords = self.extract_ragamala_keywords(combined_text)
        
        return len(keywords) > 0
    
    async def download_image(self, url: str, filename: str) -> bool:
        """Download image from URL."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Validate image
                    try:
                        img = Image.open(io.BytesIO(content))
                        if img.size[0] < 512 or img.size[1] < 512:
                            logger.warning(f"Image too small: {img.size}")
                            return False
                    except Exception as e:
                        logger.error(f"Invalid image: {e}")
                        return False
                    
                    # Save image
                    filepath = self.output_dir / filename
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    
                    logger.info(f"Downloaded: {filename}")
                    return True
                    
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
        
        return False

class MetropolitanMuseumCollector(MuseumAPICollector):
    """Collector for Metropolitan Museum of Art API."""
    
    def __init__(self, output_dir: str = "data/raw/metropolitan"):
        config = MuseumConfig(
            name="Metropolitan Museum",
            base_url="https://collectionapi.metmuseum.org/public/collection/v1",
            rate_limit=0.5
        )
        super().__init__(config, output_dir)
    
    async def search_ragamala(self, query: str = "ragamala") -> List[int]:
        """Search for Ragamala paintings."""
        search_url = f"{self.config.base_url}/search"
        params = {
            'q': query,
            'hasImages': 'true',
            'medium': 'Paintings',
            'geoLocation': 'India'
        }
        
        response = await self.make_request(search_url, params)
        if response and 'objectIDs' in response:
            logger.info(f"Found {len(response['objectIDs'])} objects for '{query}'")
            return response['objectIDs'][:100]  # Limit results
        
        return []
    
    async def get_object_details(self, object_id: int) -> Optional[ArtworkMetadata]:
        """Get detailed information for an object."""
        object_url = f"{self.config.base_url}/objects/{object_id}"
        response = await self.make_request(object_url)
        
        if not response:
            return None
        
        # Check if Ragamala-related
        if not self.is_ragamala_related(response):
            return None
        
        # Extract metadata
        try:
            metadata = ArtworkMetadata(
                id=str(object_id),
                title=response.get('title', ''),
                artist=response.get('artistDisplayName', ''),
                date=response.get('objectDate', ''),
                period=response.get('period', ''),
                style=self.extract_style(response),
                raga=self.extract_raga(response),
                region=response.get('geography', ''),
                medium=response.get('medium', ''),
                dimensions=response.get('dimensions', ''),
                description=response.get('objectName', ''),
                keywords=self.extract_ragamala_keywords(
                    f"{response.get('title', '')} {response.get('tags', '')}"
                ),
                museum="Metropolitan Museum",
                collection=response.get('department', ''),
                image_url=response.get('primaryImage', ''),
                thumbnail_url=response.get('primaryImageSmall', ''),
                rights=response.get('rightsAndReproduction', ''),
                source_url=response.get('objectURL', ''),
                collected_date=datetime.now().isoformat(),
                quality_score=self.calculate_quality_score(response)
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing object {object_id}: {e}")
            return None
    
    def extract_style(self, data: Dict) -> Optional[str]:
        """Extract painting style from metadata."""
        culture = data.get('culture', '').lower()
        classification = data.get('classification', '').lower()
        
        if 'rajput' in culture or 'rajasthani' in culture:
            return 'rajput'
        elif 'pahari' in culture or 'kangra' in culture or 'basohli' in culture:
            return 'pahari'
        elif 'deccan' in culture or 'golconda' in culture or 'bijapur' in culture:
            return 'deccan'
        elif 'mughal' in culture:
            return 'mughal'
        
        return None
    
    def extract_raga(self, data: Dict) -> Optional[str]:
        """Extract raga name from metadata."""
        title = data.get('title', '').lower()
        description = data.get('objectName', '').lower()
        
        raga_names = [
            'bhairav', 'malkauns', 'yaman', 'darbari', 'bageshri',
            'todi', 'puriya', 'marwa', 'bhimpalasi', 'kafi'
        ]
        
        for raga in raga_names:
            if raga in title or raga in description:
                return raga
        
        return None
    
    def calculate_quality_score(self, data: Dict) -> float:
        """Calculate quality score for artwork."""
        score = 0.0
        
        # Image quality
        if data.get('primaryImage'):
            score += 0.3
        
        # Metadata completeness
        if data.get('title'):
            score += 0.2
        if data.get('artistDisplayName'):
            score += 0.1
        if data.get('objectDate'):
            score += 0.1
        if data.get('culture'):
            score += 0.1
        
        # Ragamala relevance
        keywords = self.extract_ragamala_keywords(
            f"{data.get('title', '')} {data.get('objectName', '')}"
        )
        score += min(len(keywords) * 0.05, 0.2)
        
        return min(score, 1.0)

class BritishMuseumCollector(MuseumAPICollector):
    """Collector for British Museum API."""
    
    def __init__(self, output_dir: str = "data/raw/british_museum"):
        config = MuseumConfig(
            name="British Museum",
            base_url="https://www.britishmuseum.org/api/v1",
            rate_limit=1.0
        )
        super().__init__(config, output_dir)
    
    async def search_ragamala(self) -> List[Dict]:
        """Search for Ragamala paintings in British Museum."""
        search_url = f"{self.config.base_url}/collection/search"
        params = {
            'query': 'ragamala OR raga OR ragini',
            'format': 'json',
            'limit': 100
        }
        
        response = await self.make_request(search_url, params)
        if response and 'results' in response:
            return response['results']
        
        return []

class VictoriaAlbertCollector(MuseumAPICollector):
    """Collector for Victoria and Albert Museum API."""
    
    def __init__(self, api_key: str, output_dir: str = "data/raw/victoria_albert"):
        config = MuseumConfig(
            name="Victoria and Albert Museum",
            base_url="https://api.vam.ac.uk/v2",
            api_key=api_key,
            rate_limit=0.5
        )
        super().__init__(config, output_dir)
    
    async def search_ragamala(self) -> List[Dict]:
        """Search for Ragamala paintings in V&A collection."""
        search_url = f"{self.config.base_url}/objects/search"
        params = {
            'q': 'ragamala',
            'images_exist': 1,
            'page_size': 100
        }
        
        response = await self.make_request(search_url, params)
        if response and 'records' in response:
            return response['records']
        
        return []

class IndianMuseumsCollector(MuseumAPICollector):
    """Collector for Indian Museums portal."""
    
    def __init__(self, output_dir: str = "data/raw/indian_museums"):
        config = MuseumConfig(
            name="Museums of India",
            base_url="https://www.museumsofindia.gov.in/api",
            rate_limit=2.0
        )
        super().__init__(config, output_dir)
    
    async def collect_from_csv(self, csv_file: str) -> List[ArtworkMetadata]:
        """Collect data from pre-scraped CSV file."""
        artworks = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self.is_ragamala_related(row):
                        metadata = self.csv_row_to_metadata(row)
                        if metadata:
                            artworks.append(metadata)
        
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file}: {e}")
        
        return artworks
    
    def csv_row_to_metadata(self, row: Dict) -> Optional[ArtworkMetadata]:
        """Convert CSV row to ArtworkMetadata."""
        try:
            return ArtworkMetadata(
                id=row.get('id', ''),
                title=row.get('title', ''),
                artist=row.get('artist', ''),
                date=row.get('date', ''),
                period=row.get('period', ''),
                style=row.get('style', ''),
                raga=row.get('raga', ''),
                region=row.get('region', ''),
                medium=row.get('medium', ''),
                dimensions=row.get('dimensions', ''),
                description=row.get('description', ''),
                keywords=row.get('keywords', '').split(',') if row.get('keywords') else [],
                museum=row.get('museum', 'Indian Museums'),
                collection=row.get('collection', ''),
                image_url=row.get('image_url', ''),
                thumbnail_url=row.get('thumbnail_url', ''),
                rights=row.get('rights', ''),
                source_url=row.get('source_url', ''),
                collected_date=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error converting CSV row: {e}")
            return None

class RagamalaDataCollector:
    """Main collector orchestrating multiple museum sources."""
    
    def __init__(self, config_file: str = "config/collector_config.json"):
        self.config = self.load_config(config_file)
        self.output_dir = Path(self.config.get('output_dir', 'data/raw'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.s3_manager = S3Manager() if self.config.get('use_s3') else None
        
    def load_config(self, config_file: str) -> Dict:
        """Load collector configuration."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return {
                'output_dir': 'data/raw',
                'use_s3': False,
                'max_images_per_source': 100,
                'min_quality_score': 0.5
            }
    
    async def collect_all_sources(self) -> Dict[str, List[ArtworkMetadata]]:
        """Collect from all configured sources."""
        results = {}
        
        # Metropolitan Museum
        if self.config.get('collect_met', True):
            logger.info("Collecting from Metropolitan Museum...")
            async with MetropolitanMuseumCollector() as collector:
                met_results = await self.collect_from_met(collector)
                results['metropolitan'] = met_results
        
        # British Museum
        if self.config.get('collect_british', True):
            logger.info("Collecting from British Museum...")
            async with BritishMuseumCollector() as collector:
                british_results = await self.collect_from_british(collector)
                results['british'] = british_results
        
        # Victoria & Albert
        if self.config.get('collect_va', True) and self.config.get('va_api_key'):
            logger.info("Collecting from Victoria & Albert Museum...")
            async with VictoriaAlbertCollector(self.config['va_api_key']) as collector:
                va_results = await self.collect_from_va(collector)
                results['victoria_albert'] = va_results
        
        # Indian Museums
        if self.config.get('collect_indian', True):
            logger.info("Collecting from Indian Museums...")
            indian_collector = IndianMuseumsCollector()
            indian_results = await self.collect_from_indian(indian_collector)
            results['indian_museums'] = indian_results
        
        return results
    
    async def collect_from_met(self, collector: MetropolitanMuseumCollector) -> List[ArtworkMetadata]:
        """Collect from Metropolitan Museum."""
        artworks = []
        
        # Search for different terms
        search_terms = ['ragamala', 'raga', 'ragini', 'indian miniature']
        
        for term in search_terms:
            object_ids = await collector.search_ragamala(term)
            
            for object_id in object_ids[:self.config.get('max_images_per_source', 100)]:
                metadata = await collector.get_object_details(object_id)
                if metadata and metadata.quality_score >= self.config.get('min_quality_score', 0.5):
                    artworks.append(metadata)
                    
                    # Download image
                    if metadata.image_url:
                        filename = f"met_{metadata.id}.jpg"
                        await collector.download_image(metadata.image_url, filename)
        
        return artworks
    
    async def collect_from_british(self, collector: BritishMuseumCollector) -> List[ArtworkMetadata]:
        """Collect from British Museum."""
        # Implementation for British Museum API
        return []
    
    async def collect_from_va(self, collector: VictoriaAlbertCollector) -> List[ArtworkMetadata]:
        """Collect from Victoria & Albert Museum."""
        # Implementation for V&A API
        return []
    
    async def collect_from_indian(self, collector: IndianMuseumsCollector) -> List[ArtworkMetadata]:
        """Collect from Indian Museums."""
        # Use pre-scraped CSV data
        csv_files = [
            'data/external/indian_museums_ragamala.csv',
            'data/external/national_museum_delhi.csv',
            'data/external/prince_wales_mumbai.csv'
        ]
        
        artworks = []
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                csv_artworks = await collector.collect_from_csv(csv_file)
                artworks.extend(csv_artworks)
        
        return artworks
    
    def save_metadata(self, results: Dict[str, List[ArtworkMetadata]]):
        """Save collected metadata to files."""
        all_metadata = []
        
        for source, artworks in results.items():
            logger.info(f"Collected {len(artworks)} artworks from {source}")
            
            # Save source-specific metadata
            source_file = self.output_dir / f"{source}_metadata.json"
            with open(source_file, 'w', encoding='utf-8') as f:
                json.dump([artwork.to_dict() for artwork in artworks], f, indent=2, ensure_ascii=False)
            
            all_metadata.extend(artworks)
        
        # Save combined metadata
        combined_file = self.output_dir / "combined_metadata.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump([artwork.to_dict() for artwork in all_metadata], f, indent=2, ensure_ascii=False)
        
        # Generate JSONL for training
        jsonl_file = self.output_dir.parent / "metadata" / "metadata.jsonl"
        jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for artwork in all_metadata:
                if artwork.image_url and artwork.quality_score >= self.config.get('min_quality_score', 0.5):
                    training_record = {
                        'file_name': f"{artwork.museum.lower().replace(' ', '_')}_{artwork.id}.jpg",
                        'text': self.generate_training_prompt(artwork),
                        'raga': artwork.raga,
                        'style': artwork.style,
                        'period': artwork.period,
                        'region': artwork.region,
                        'quality_score': artwork.quality_score
                    }
                    f.write(json.dumps(training_record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(all_metadata)} total artworks")
    
    def generate_training_prompt(self, artwork: ArtworkMetadata) -> str:
        """Generate training prompt for artwork."""
        prompt_parts = []
        
        if artwork.style:
            prompt_parts.append(f"A {artwork.style} style")
        
        prompt_parts.append("ragamala painting")
        
        if artwork.raga:
            prompt_parts.append(f"depicting raga {artwork.raga}")
        
        if artwork.description:
            prompt_parts.append(f"showing {artwork.description.lower()}")
        
        if artwork.period:
            prompt_parts.append(f"from {artwork.period}")
        
        return " ".join(prompt_parts)
    
    async def run_collection(self):
        """Run the complete collection process."""
        logger.info("Starting Ragamala data collection...")
        
        try:
            results = await self.collect_all_sources()
            self.save_metadata(results)
            
            if self.s3_manager:
                logger.info("Uploading to S3...")
                await self.upload_to_s3(results)
            
            logger.info("Collection completed successfully!")
            
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            raise
    
    async def upload_to_s3(self, results: Dict[str, List[ArtworkMetadata]]):
        """Upload collected data to S3."""
        for source, artworks in results.items():
            for artwork in artworks:
                if artwork.image_url:
                    local_file = self.output_dir / f"{source}_{artwork.id}.jpg"
                    if local_file.exists():
                        s3_key = f"ragamala/raw/{source}/{artwork.id}.jpg"
                        self.s3_manager.upload_file(str(local_file), s3_key)

async def main():
    """Main function for running the collector."""
    collector = RagamalaDataCollector()
    await collector.run_collection()

if __name__ == "__main__":
    asyncio.run(main())
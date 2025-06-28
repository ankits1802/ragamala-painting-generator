"""
Data Acquisition Script for Ragamala Painting Generation.

This script provides comprehensive data acquisition functionality for collecting
Ragamala paintings from various sources including museum APIs, digital archives,
and cultural repositories for SDXL fine-tuning.
"""

import os
import sys
import json
import time
import logging
import requests
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import mimetypes
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.aws_utils import S3Manager, AWSConfig

logger = setup_logger(__name__)

@dataclass
class DownloadConfig:
    """Configuration for data downloading."""
    # Output directories
    output_dir: str = "data/raw"
    metadata_dir: str = "data/metadata"
    
    # Download settings
    max_concurrent_downloads: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Image filtering
    min_image_size: Tuple[int, int] = (512, 512)
    max_image_size: Tuple[int, int] = (4096, 4096)
    allowed_formats: List[str] = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
    
    # API settings
    api_rate_limit: float = 1.0  # seconds between requests
    user_agent: str = "RagamalaDataCollector/1.0"
    
    # S3 settings
    enable_s3_backup: bool = False
    s3_bucket: Optional[str] = None
    s3_prefix: str = "ragamala/raw_data/"

@dataclass
class ImageMetadata:
    """Metadata for downloaded images."""
    filename: str
    url: str
    source: str
    title: str
    artist: Optional[str]
    period: Optional[str]
    style: Optional[str]
    raga: Optional[str]
    description: Optional[str]
    dimensions: Tuple[int, int]
    file_size: int
    download_date: str
    cultural_tags: List[str]
    museum_id: Optional[str]

class MuseumAPIClient:
    """Base client for museum APIs."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent,
            'Accept': 'application/json'
        })
    
    async def get_async(self, url: str, **kwargs) -> Optional[Dict]:
        """Async GET request with error handling."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=self.config.request_timeout, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
    
    def get_sync(self, url: str, **kwargs) -> Optional[Dict]:
        """Synchronous GET request with error handling."""
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.get(url, timeout=self.config.request_timeout, **kwargs)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    return None
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
        
        return None

class MetMuseumClient(MuseumAPIClient):
    """Client for Metropolitan Museum of Art API."""
    
    BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
    
    def search_ragamala_paintings(self) -> List[int]:
        """Search for Ragamala paintings in Met collection."""
        search_terms = [
            "ragamala", "raga", "indian miniature", "rajput painting",
            "pahari painting", "mughal painting", "deccan painting"
        ]
        
        object_ids = set()
        
        for term in search_terms:
            logger.info(f"Searching Met Museum for: {term}")
            
            url = f"{self.BASE_URL}/search"
            params = {
                'q': term,
                'hasImages': 'true',
                'medium': 'Paintings',
                'geoLocation': 'India'
            }
            
            data = self.get_sync(url, params=params)
            if data and 'objectIDs' in data:
                object_ids.update(data['objectIDs'])
                logger.info(f"Found {len(data['objectIDs'])} objects for '{term}'")
            
            time.sleep(self.config.api_rate_limit)
        
        return list(object_ids)
    
    def get_object_details(self, object_id: int) -> Optional[ImageMetadata]:
        """Get detailed information about a specific object."""
        url = f"{self.BASE_URL}/objects/{object_id}"
        data = self.get_sync(url)
        
        if not data:
            return None
        
        # Check if it's relevant to Ragamala
        if not self._is_ragamala_relevant(data):
            return None
        
        # Extract image URL
        image_url = data.get('primaryImage') or data.get('primaryImageSmall')
        if not image_url:
            return None
        
        # Extract metadata
        title = data.get('title', 'Untitled')
        artist = data.get('artistDisplayName', 'Unknown')
        period = data.get('period', data.get('dynasty'))
        culture = data.get('culture', '')
        
        # Determine style and raga from metadata
        style = self._extract_style(data)
        raga = self._extract_raga(data)
        
        # Create metadata
        metadata = ImageMetadata(
            filename=f"met_{object_id}.jpg",
            url=image_url,
            source="Metropolitan Museum",
            title=title,
            artist=artist,
            period=period,
            style=style,
            raga=raga,
            description=data.get('medium', ''),
            dimensions=(0, 0),  # Will be updated after download
            file_size=0,
            download_date=time.strftime("%Y-%m-%d"),
            cultural_tags=self._extract_cultural_tags(data),
            museum_id=str(object_id)
        )
        
        return metadata
    
    def _is_ragamala_relevant(self, data: Dict) -> bool:
        """Check if the object is relevant to Ragamala paintings."""
        text_fields = [
            data.get('title', '').lower(),
            data.get('medium', '').lower(),
            data.get('classification', '').lower(),
            data.get('culture', '').lower(),
            ' '.join(data.get('tags', [])).lower()
        ]
        
        ragamala_keywords = [
            'ragamala', 'raga', 'miniature', 'rajput', 'pahari',
            'mughal', 'deccan', 'indian painting', 'folio'
        ]
        
        text_content = ' '.join(text_fields)
        return any(keyword in text_content for keyword in ragamala_keywords)
    
    def _extract_style(self, data: Dict) -> Optional[str]:
        """Extract painting style from metadata."""
        culture = data.get('culture', '').lower()
        classification = data.get('classification', '').lower()
        title = data.get('title', '').lower()
        
        style_keywords = {
            'rajput': ['rajput', 'rajasthani', 'mewar', 'marwar'],
            'pahari': ['pahari', 'kangra', 'basohli', 'guler'],
            'mughal': ['mughal', 'imperial'],
            'deccan': ['deccan', 'deccani', 'golconda', 'bijapur']
        }
        
        text_content = f"{culture} {classification} {title}"
        
        for style, keywords in style_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                return style
        
        return None
    
    def _extract_raga(self, data: Dict) -> Optional[str]:
        """Extract raga name from metadata."""
        title = data.get('title', '').lower()
        
        raga_names = [
            'bhairav', 'yaman', 'malkauns', 'darbari', 'bageshri',
            'todi', 'puriya', 'marwa', 'bhimpalasi', 'kafi'
        ]
        
        for raga in raga_names:
            if raga in title:
                return raga
        
        return None
    
    def _extract_cultural_tags(self, data: Dict) -> List[str]:
        """Extract cultural tags from metadata."""
        tags = []
        
        if data.get('culture'):
            tags.append(data['culture'])
        
        if data.get('period'):
            tags.append(data['period'])
        
        if data.get('dynasty'):
            tags.append(data['dynasty'])
        
        # Add predefined tags
        existing_tags = data.get('tags', [])
        tags.extend(existing_tags)
        
        return list(set(tags))

class WikiArtClient(MuseumAPIClient):
    """Client for WikiArt scraping."""
    
    BASE_URL = "https://www.wikiart.org"
    
    def search_indian_miniatures(self) -> List[str]:
        """Search for Indian miniature paintings on WikiArt."""
        search_urls = [
            f"{self.BASE_URL}/en/paintings-by-style/miniature",
            f"{self.BASE_URL}/en/paintings-by-genre/religious-painting",
            f"{self.BASE_URL}/en/artists-by-nation/indian"
        ]
        
        painting_urls = []
        
        for url in search_urls:
            logger.info(f"Scraping WikiArt: {url}")
            
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    # Parse HTML to extract painting URLs
                    # This would require BeautifulSoup for proper implementation
                    # For now, return placeholder
                    pass
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
            
            time.sleep(self.config.api_rate_limit)
        
        return painting_urls

class DigitalLibraryClient(MuseumAPIClient):
    """Client for digital library collections."""
    
    def __init__(self, config: DownloadConfig):
        super().__init__(config)
        self.sources = {
            'british_museum': 'https://www.britishmuseum.org/api/collection',
            'v_and_a': 'https://api.vam.ac.uk/v2/objects/search',
            'lacma': 'https://api.lacma.org/collection',
            'cleveland': 'https://openaccess-api.clevelandart.org/api/artworks'
        }
    
    def search_all_sources(self) -> List[ImageMetadata]:
        """Search all digital library sources."""
        all_metadata = []
        
        for source_name, base_url in self.sources.items():
            logger.info(f"Searching {source_name}...")
            
            try:
                metadata_list = self._search_source(source_name, base_url)
                all_metadata.extend(metadata_list)
                logger.info(f"Found {len(metadata_list)} items from {source_name}")
            except Exception as e:
                logger.error(f"Error searching {source_name}: {e}")
        
        return all_metadata
    
    def _search_source(self, source_name: str, base_url: str) -> List[ImageMetadata]:
        """Search a specific digital library source."""
        # Implementation would vary by API
        # This is a placeholder for the actual implementation
        return []

class ImageDownloader:
    """Handles image downloading and processing."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.s3_manager = None
        
        if config.enable_s3_backup and config.s3_bucket:
            aws_config = AWSConfig(s3_bucket_name=config.s3_bucket)
            self.s3_manager = S3Manager(aws_config)
    
    async def download_image_async(self, metadata: ImageMetadata) -> bool:
        """Download image asynchronously."""
        output_path = Path(self.config.output_dir) / metadata.filename
        
        # Check if already exists
        if output_path.exists():
            logger.info(f"Image already exists: {metadata.filename}")
            return True
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(metadata.url, timeout=self.config.request_timeout) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Validate image
                        if not self._validate_image_content(content):
                            logger.warning(f"Invalid image content: {metadata.url}")
                            return False
                        
                        # Save image
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, 'wb') as f:
                            f.write(content)
                        
                        # Update metadata with actual dimensions
                        self._update_image_metadata(metadata, output_path)
                        
                        # Backup to S3 if enabled
                        if self.s3_manager:
                            s3_key = f"{self.config.s3_prefix}{metadata.filename}"
                            self.s3_manager.upload_file(output_path, s3_key)
                        
                        logger.info(f"Downloaded: {metadata.filename}")
                        return True
                    else:
                        logger.warning(f"HTTP {response.status} for {metadata.url}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error downloading {metadata.url}: {e}")
            return False
    
    def download_image_sync(self, metadata: ImageMetadata) -> bool:
        """Download image synchronously."""
        output_path = Path(self.config.output_dir) / metadata.filename
        
        # Check if already exists
        if output_path.exists():
            logger.info(f"Image already exists: {metadata.filename}")
            return True
        
        try:
            response = requests.get(metadata.url, timeout=self.config.request_timeout, stream=True)
            if response.status_code == 200:
                content = response.content
                
                # Validate image
                if not self._validate_image_content(content):
                    logger.warning(f"Invalid image content: {metadata.url}")
                    return False
                
                # Save image
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(content)
                
                # Update metadata with actual dimensions
                self._update_image_metadata(metadata, output_path)
                
                # Backup to S3 if enabled
                if self.s3_manager:
                    s3_key = f"{self.config.s3_prefix}{metadata.filename}"
                    self.s3_manager.upload_file(output_path, s3_key)
                
                logger.info(f"Downloaded: {metadata.filename}")
                return True
            else:
                logger.warning(f"HTTP {response.status_code} for {metadata.url}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {metadata.url}: {e}")
            return False
    
    def _validate_image_content(self, content: bytes) -> bool:
        """Validate image content."""
        try:
            # Check if it's a valid image
            image = Image.open(io.BytesIO(content))
            
            # Check dimensions
            width, height = image.size
            min_w, min_h = self.config.min_image_size
            max_w, max_h = self.config.max_image_size
            
            if width < min_w or height < min_h:
                logger.warning(f"Image too small: {width}x{height}")
                return False
            
            if width > max_w or height > max_h:
                logger.warning(f"Image too large: {width}x{height}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False
    
    def _update_image_metadata(self, metadata: ImageMetadata, image_path: Path):
        """Update metadata with actual image information."""
        try:
            with Image.open(image_path) as img:
                metadata.dimensions = img.size
            
            metadata.file_size = image_path.stat().st_size
            
        except Exception as e:
            logger.error(f"Error updating metadata for {image_path}: {e}")
    
    async def download_batch_async(self, metadata_list: List[ImageMetadata]) -> Dict[str, bool]:
        """Download multiple images asynchronously."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_downloads)
        
        async def download_with_semaphore(metadata):
            async with semaphore:
                return await self.download_image_async(metadata)
        
        tasks = [download_with_semaphore(metadata) for metadata in metadata_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Create results dictionary
        download_results = {}
        for metadata, result in zip(metadata_list, results):
            if isinstance(result, Exception):
                logger.error(f"Exception downloading {metadata.filename}: {result}")
                download_results[metadata.filename] = False
            else:
                download_results[metadata.filename] = result
        
        return download_results
    
    def download_batch_sync(self, metadata_list: List[ImageMetadata]) -> Dict[str, bool]:
        """Download multiple images using thread pool."""
        download_results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_downloads) as executor:
            future_to_metadata = {
                executor.submit(self.download_image_sync, metadata): metadata
                for metadata in metadata_list
            }
            
            for future in tqdm(as_completed(future_to_metadata), total=len(metadata_list)):
                metadata = future_to_metadata[future]
                try:
                    result = future.result()
                    download_results[metadata.filename] = result
                except Exception as e:
                    logger.error(f"Exception downloading {metadata.filename}: {e}")
                    download_results[metadata.filename] = False
        
        return download_results

class MetadataManager:
    """Manages metadata storage and retrieval."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.metadata_file = Path(config.metadata_dir) / "downloaded_images.jsonl"
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_metadata(self, metadata_list: List[ImageMetadata]):
        """Save metadata to JSONL file."""
        with open(self.metadata_file, 'a', encoding='utf-8') as f:
            for metadata in metadata_list:
                json_line = json.dumps(asdict(metadata), ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"Saved metadata for {len(metadata_list)} images")
    
    def load_metadata(self) -> List[ImageMetadata]:
        """Load metadata from JSONL file."""
        metadata_list = []
        
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        metadata = ImageMetadata(**data)
                        metadata_list.append(metadata)
                    except Exception as e:
                        logger.error(f"Error parsing metadata line: {e}")
        
        return metadata_list
    
    def get_downloaded_filenames(self) -> set:
        """Get set of already downloaded filenames."""
        metadata_list = self.load_metadata()
        return {metadata.filename for metadata in metadata_list}
    
    def create_summary_report(self) -> Dict[str, Any]:
        """Create summary report of downloaded data."""
        metadata_list = self.load_metadata()
        
        if not metadata_list:
            return {"total_images": 0}
        
        # Calculate statistics
        total_images = len(metadata_list)
        sources = {}
        styles = {}
        ragas = {}
        periods = {}
        
        total_size = 0
        
        for metadata in metadata_list:
            # Count by source
            sources[metadata.source] = sources.get(metadata.source, 0) + 1
            
            # Count by style
            if metadata.style:
                styles[metadata.style] = styles.get(metadata.style, 0) + 1
            
            # Count by raga
            if metadata.raga:
                ragas[metadata.raga] = ragas.get(metadata.raga, 0) + 1
            
            # Count by period
            if metadata.period:
                periods[metadata.period] = periods.get(metadata.period, 0) + 1
            
            total_size += metadata.file_size
        
        return {
            "total_images": total_images,
            "total_size_mb": total_size / (1024 * 1024),
            "sources": sources,
            "styles": styles,
            "ragas": ragas,
            "periods": periods
        }

class RagamalaDataCollector:
    """Main data collection orchestrator."""
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        self.config = config or DownloadConfig()
        self.met_client = MetMuseumClient(self.config)
        self.wikiart_client = WikiArtClient(self.config)
        self.digital_library_client = DigitalLibraryClient(self.config)
        self.downloader = ImageDownloader(self.config)
        self.metadata_manager = MetadataManager(self.config)
    
    async def collect_all_data(self) -> Dict[str, Any]:
        """Collect data from all sources."""
        logger.info("Starting comprehensive data collection...")
        
        all_metadata = []
        
        # Collect from Met Museum
        logger.info("Collecting from Metropolitan Museum...")
        met_object_ids = self.met_client.search_ragamala_paintings()
        
        for object_id in tqdm(met_object_ids, desc="Processing Met objects"):
            metadata = self.met_client.get_object_details(object_id)
            if metadata:
                all_metadata.append(metadata)
            time.sleep(self.config.api_rate_limit)
        
        # Collect from digital libraries
        logger.info("Collecting from digital libraries...")
        digital_metadata = self.digital_library_client.search_all_sources()
        all_metadata.extend(digital_metadata)
        
        # Filter out already downloaded images
        existing_filenames = self.metadata_manager.get_downloaded_filenames()
        new_metadata = [m for m in all_metadata if m.filename not in existing_filenames]
        
        logger.info(f"Found {len(new_metadata)} new images to download")
        
        # Download images
        if new_metadata:
            download_results = await self.downloader.download_batch_async(new_metadata)
            
            # Filter successful downloads
            successful_metadata = [
                metadata for metadata in new_metadata
                if download_results.get(metadata.filename, False)
            ]
            
            # Save metadata
            self.metadata_manager.save_metadata(successful_metadata)
            
            logger.info(f"Successfully downloaded {len(successful_metadata)} images")
        
        # Generate summary report
        summary = self.metadata_manager.create_summary_report()
        
        return {
            "summary": summary,
            "new_downloads": len(new_metadata),
            "successful_downloads": len([r for r in download_results.values() if r]) if new_metadata else 0
        }
    
    def collect_sync(self) -> Dict[str, Any]:
        """Synchronous version of data collection."""
        logger.info("Starting synchronous data collection...")
        
        all_metadata = []
        
        # Collect from Met Museum
        logger.info("Collecting from Metropolitan Museum...")
        met_object_ids = self.met_client.search_ragamala_paintings()
        
        for object_id in tqdm(met_object_ids, desc="Processing Met objects"):
            metadata = self.met_client.get_object_details(object_id)
            if metadata:
                all_metadata.append(metadata)
            time.sleep(self.config.api_rate_limit)
        
        # Filter out already downloaded images
        existing_filenames = self.metadata_manager.get_downloaded_filenames()
        new_metadata = [m for m in all_metadata if m.filename not in existing_filenames]
        
        logger.info(f"Found {len(new_metadata)} new images to download")
        
        # Download images
        if new_metadata:
            download_results = self.downloader.download_batch_sync(new_metadata)
            
            # Filter successful downloads
            successful_metadata = [
                metadata for metadata in new_metadata
                if download_results.get(metadata.filename, False)
            ]
            
            # Save metadata
            self.metadata_manager.save_metadata(successful_metadata)
            
            logger.info(f"Successfully downloaded {len(successful_metadata)} images")
        
        # Generate summary report
        summary = self.metadata_manager.create_summary_report()
        
        return {
            "summary": summary,
            "new_downloads": len(new_metadata),
            "successful_downloads": len([r for r in download_results.values() if r]) if new_metadata else 0
        }

def main():
    """Main function for data acquisition."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Ragamala painting data")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent downloads")
    parser.add_argument("--async-mode", action="store_true", help="Use async downloading")
    parser.add_argument("--s3-backup", action="store_true", help="Enable S3 backup")
    parser.add_argument("--s3-bucket", help="S3 bucket name")
    
    args = parser.parse_args()
    
    # Create configuration
    config = DownloadConfig(
        output_dir=args.output_dir,
        max_concurrent_downloads=args.max_concurrent,
        enable_s3_backup=args.s3_backup,
        s3_bucket=args.s3_bucket
    )
    
    # Create collector
    collector = RagamalaDataCollector(config)
    
    # Run collection
    if args.async_mode:
        import asyncio
        result = asyncio.run(collector.collect_all_data())
    else:
        result = collector.collect_sync()
    
    # Print results
    print("\nData Collection Results:")
    print(f"New downloads: {result['new_downloads']}")
    print(f"Successful downloads: {result['successful_downloads']}")
    print(f"Total images in collection: {result['summary']['total_images']}")
    print(f"Total size: {result['summary']['total_size_mb']:.2f} MB")
    
    if result['summary']['styles']:
        print("\nStyles distribution:")
        for style, count in result['summary']['styles'].items():
            print(f"  {style}: {count}")
    
    if result['summary']['ragas']:
        print("\nRagas distribution:")
        for raga, count in result['summary']['ragas'].items():
            print(f"  {raga}: {count}")

if __name__ == "__main__":
    main()

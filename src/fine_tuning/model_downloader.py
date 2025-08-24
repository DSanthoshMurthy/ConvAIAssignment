import os
import gdown
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self):
        # Create cache directory if it doesn't exist
        self.cache_dir = Path.home() / '.financial_rag_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # File to keep track of downloaded models
        self.downloaded_files = self.cache_dir / 'downloaded_files.txt'
        if not self.downloaded_files.exists():
            self.downloaded_files.touch()
    
    def is_file_downloaded(self, file_id):
        """Check if file was already downloaded"""
        if self.downloaded_files.exists():
            with open(self.downloaded_files, 'r') as f:
                return file_id in f.read()
        return False
    
    def mark_as_downloaded(self, file_id):
        """Mark file as downloaded"""
        with open(self.downloaded_files, 'a') as f:
            f.write(f"{file_id}\n")
    
    def download_model(self, file_id, output_path=None):
        """
        Download model from Google Drive
        
        Args:
            file_id: Google Drive file ID
            output_path: Where to save the model. If None, saves to cache directory
        
        Returns:
            Path to downloaded file
        """
        try:
            if output_path is None:
                output_path = self.cache_dir / f"model_{file_id}.pt"
            
            # Convert to string if it's a Path object
            output_path = str(output_path)
            
            # Check if already downloaded
            if self.is_file_downloaded(file_id) and os.path.exists(output_path):
                logger.info(f"Model already downloaded at {output_path}")
                return output_path
            
            # Create Google Drive URL
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # Download file
            logger.info(f"Downloading model from Google Drive...")
            gdown.download(url, output_path, quiet=False)
            
            # Verify download
            if os.path.exists(output_path):
                self.mark_as_downloaded(file_id)
                logger.info(f"Model downloaded successfully to {output_path}")
                return output_path
            else:
                raise Exception("Download failed: File not found after download")
                
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
    
    def clean_cache(self):
        """Remove all cached models"""
        try:
            for file in self.cache_dir.glob("model_*.pt"):
                file.unlink()
            self.downloaded_files.unlink(missing_ok=True)
            logger.info("Cache cleaned successfully")
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
            raise

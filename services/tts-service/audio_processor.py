import os
import asyncio
from pathlib import Path
from typing import Optional
from pydub import AudioSegment
from pydub.effects import normalize
from shared.logging import get_logger

logger = get_logger(__name__)

class AudioProcessor:
    """Audio processing utilities for TTS service"""
    
    def __init__(self):
        # Ensure ffmpeg is available for pydub
        pass
    
    async def get_duration(self, file_path: str) -> float:
        """Get audio file duration in seconds"""
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0
    
    async def adjust_volume(self, audio_data: bytes, volume_multiplier: float) -> bytes:
        """Adjust audio volume"""
        try:
            # Save temporary file
            temp_path = "/tmp/temp_audio_input.mp3"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # Load and adjust volume
            audio = AudioSegment.from_file(temp_path)
            
            # Convert volume multiplier to dB
            db_change = 20 * (volume_multiplier - 1)  # Approximate conversion
            adjusted_audio = audio + db_change
            
            # Export back to bytes
            output_path = "/tmp/temp_audio_output.mp3"
            adjusted_audio.export(output_path, format="mp3")
            
            with open(output_path, 'rb') as f:
                result = f.read()
            
            # Cleanup
            os.remove(temp_path)
            os.remove(output_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to adjust volume: {e}")
            return audio_data  # Return original on error
    
    async def normalize_audio(self, file_path: str) -> str:
        """Normalize audio levels"""
        try:
            audio = AudioSegment.from_file(file_path)
            normalized_audio = normalize(audio)
            
            # Create output path
            path_obj = Path(file_path)
            output_path = path_obj.parent / f"{path_obj.stem}_normalized{path_obj.suffix}"
            
            normalized_audio.export(str(output_path), format="mp3")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to normalize audio: {e}")
            return file_path  # Return original on error
    
    async def trim_silence(self, file_path: str, silence_thresh: int = -50) -> str:
        """Trim silence from beginning and end of audio"""
        try:
            audio = AudioSegment.from_file(file_path)
            
            # Trim silence
            trimmed_audio = audio.strip_silence(silence_thresh=silence_thresh)
            
            # Create output path
            path_obj = Path(file_path)
            output_path = path_obj.parent / f"{path_obj.stem}_trimmed{path_obj.suffix}"
            
            trimmed_audio.export(str(output_path), format="mp3")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to trim silence: {e}")
            return file_path  # Return original on error
    
    async def adjust_speed(self, file_path: str, speed_multiplier: float) -> str:
        """Adjust audio playback speed"""
        try:
            audio = AudioSegment.from_file(file_path)
            
            # Change speed by changing frame rate
            new_sample_rate = int(audio.frame_rate * speed_multiplier)
            speed_adjusted = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
            
            # Resample back to original rate to maintain pitch
            speed_adjusted = speed_adjusted.set_frame_rate(audio.frame_rate)
            
            # Create output path
            path_obj = Path(file_path)
            output_path = path_obj.parent / f"{path_obj.stem}_speed{speed_multiplier}{path_obj.suffix}"
            
            speed_adjusted.export(str(output_path), format="mp3")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to adjust speed: {e}")
            return file_path  # Return original on error
    
    async def convert_format(self, file_path: str, target_format: str) -> str:
        """Convert audio to different format"""
        try:
            audio = AudioSegment.from_file(file_path)
            
            # Create output path with new extension
            path_obj = Path(file_path)
            output_path = path_obj.parent / f"{path_obj.stem}.{target_format}"
            
            # Export in target format
            audio.export(str(output_path), format=target_format)
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to convert format: {e}")
            return file_path  # Return original on error
    
    async def merge_audio_files(self, file_paths: list, output_path: str) -> str:
        """Merge multiple audio files into one"""
        try:
            combined = AudioSegment.empty()
            
            for file_path in file_paths:
                audio = AudioSegment.from_file(file_path)
                combined += audio
            
            combined.export(output_path, format="mp3")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to merge audio files: {e}")
            raise
    
    async def add_fade(self, file_path: str, fade_in_ms: int = 0, fade_out_ms: int = 0) -> str:
        """Add fade in/out effects to audio"""
        try:
            audio = AudioSegment.from_file(file_path)
            
            if fade_in_ms > 0:
                audio = audio.fade_in(fade_in_ms)
            
            if fade_out_ms > 0:
                audio = audio.fade_out(fade_out_ms)
            
            # Create output path
            path_obj = Path(file_path)
            output_path = path_obj.parent / f"{path_obj.stem}_faded{path_obj.suffix}"
            
            audio.export(str(output_path), format="mp3")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to add fade effects: {e}")
            return file_path  # Return original on error
import yt_dlp
import torch
import whisper
import os
import re
import warnings
import logging
import unicodedata

# Suppress whisper warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

# Configure yt-dlp logging
logging.getLogger("yt_dlp").setLevel(logging.ERROR)  # Only show errors, not warnings

model = whisper.load_model("small")

def sanitize_filename(filename):
    """
    Sanitize filename to be Windows-compatible
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Remove any leading/trailing periods or spaces
    filename = filename.strip('. ')
    # Ensure filename isn't empty after sanitization
    if not filename:
        filename = 'audio'
    return filename

def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL, removing playlist parameters
    """
    # Extract video ID from different types of YouTube URLs
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', url)
    if video_id_match:
        return f'https://www.youtube.com/watch?v={video_id_match.group(1)}'
    return url

def transcribe_youtube_video(video_url):
    print("Downloading audio from YouTube...")
    
    # Clean the URL to remove playlist parameters
    clean_url = extract_video_id(video_url)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": "%(title).100s.%(ext)s",  # Limit title to 100 characters
        "quiet": True,
        "no_warnings": True,  # Suppress yt-dlp warnings
        "extract_flat": False,  # Don't extract playlist information
        "noplaylist": True,    # Download only the video, not the playlist
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First get the video info without downloading
            info = ydl.extract_info(clean_url, download=False)
            
            # Sanitize the filename before download
            title = info['title']
            safe_title = sanitize_filename(title)
            
            # Update the output template with the sanitized title
            ydl_opts['outtmpl'] = f"{safe_title}.%(ext)s"
            
            # Now download with the sanitized filename
            with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                ydl2.download([clean_url])
            
            # The actual audio file will have .mp3 extension due to the postprocessor
            audio_file = f"{safe_title}.mp3"

        print("Transcribing audio...")
        result = model.transcribe(audio_file, fp16=torch.cuda.is_available())

        # Clean up the downloaded audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)

        return result["text"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
# if __name__ == "__main__":
#     youtube_url = input("Enter YouTube URL: ")
#     transcript = transcribe_youtube_video(youtube_url)
#     if transcript:
#         print("\nTranscription:\n")
#         print(transcript)
#     else:
#         print("\nFailed to transcribe the video.")
import os
import requests
from moviepy.editor import VideoFileClip
from urllib.parse import urlparse
import yt_dlp
import uuid
import glob


class VideoProcessor:
    def __init__(self, output_dir="Audio"):
        """Initialize the VideoProcessor with an output directory for audio files."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def cleanup_audio_files(self):
        """
        Remove all existing audio and video files from the output directory
        to ensure only one audio file remains after processing.
        """
        print(f"Cleaning up existing files in {self.output_dir}...")

        # Define file patterns to clean up
        patterns = [
            "*.mp3",
            "*.wav",
            "*.m4a",
            "*.aac",
            "*.ogg",
            "*.flac",  # Audio files
            "*.mp4",
            "*.avi",
            "*.mov",
            "*.mkv",
            "*.webm",  # Video files
        ]

        files_removed = 0
        for pattern in patterns:
            files = glob.glob(os.path.join(self.output_dir, pattern))
            for file_path in files:
                try:
                    os.remove(file_path)
                    files_removed += 1
                    print(f"Removed: {os.path.basename(file_path)}")
                except OSError as e:
                    print(f"Warning: Could not remove {file_path}: {e}")

        if files_removed > 0:
            print(f"Cleaned up {files_removed} files")
        else:
            print("No files to clean up")

    def download_video(self, url):
        """
        Download a video from URL (YouTube or direct mp4 link)
        Returns the path to the downloaded video file
        """
        parsed_url = urlparse(url)

        # Generate a unique filename in the Audio folder
        video_filename = f"video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(self.output_dir, video_filename)

        # YouTube URL
        if "youtube.com" in url or "youtu.be" in url:
            print(f"Downloading YouTube video from {url}")
            try:
                # Configure yt-dlp options
                ydl_opts = {
                    "format": "best[ext=mp4]/best",  # Select the best single format
                    "outtmpl": video_path,
                    "quiet": False,
                    "no_warnings": False,
                    "progress": True,
                }

                # Download the video
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    print(f"Starting download to {video_path}...")
                    ydl.download([url])
                    print("Download completed")

                # Return the path to the downloaded file
                if os.path.exists(video_path):
                    return video_path
                else:
                    raise Exception("Download failed: File not found")

            except Exception as e:
                raise Exception(f"YouTube download failed: {e}")

        # Direct MP4 link
        elif parsed_url.path.endswith(".mp4"):
            print(f"Downloading video from direct link {url}")
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                print(f"Saving video to {video_path}")
                with open(video_path, "wb") as f:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):  # 1MB chunks
                        if chunk:
                            downloaded += len(chunk)
                            f.write(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownload progress: {percent:.1f}%", end="")
                print("\nDownload complete")
                return video_path
            else:
                raise Exception(f"Failed to download video: {response.status_code}")

        else:
            raise ValueError("URL must be a YouTube link or direct mp4 link")

    def extract_audio(self, video_path, audio_format="mp3"):
        """
        Extract audio from video file and save it to the output directory
        Returns the path to the saved audio file
        """
        print(f"Extracting audio from {video_path}")
        try:
            video = VideoFileClip(video_path)

            # Generate audio filename from video filename
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_filename = f"{base_name}.{audio_format}"
            audio_path = os.path.join(self.output_dir, audio_filename)

            # Extract audio
            print(f"Writing audio to {audio_path}")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)

            # Close video to free resources
            video.close()

            return audio_path
        except Exception as e:
            raise Exception(f"Audio extraction failed: {e}")

    def process_video(self, url, delete_video=True, cleanup_first=True):
        """
        Download video and extract audio in one step

        Args:
            url (str): URL to process
            delete_video (bool): Whether to delete the video file after extraction
            cleanup_first (bool): Whether to clean up existing files before processing

        Returns:
            str: Path to the saved audio file
        """
        # Clean up existing files first if requested
        if cleanup_first:
            self.cleanup_audio_files()

        # Download video directly to the Audio folder
        video_path = self.download_video(url)

        # Extract audio
        audio_path = self.extract_audio(video_path)

        # Delete the video file if requested
        if delete_video and os.path.exists(video_path):
            os.remove(video_path)
            print(f"Deleted video file: {video_path}")

        return audio_path


# if __name__ == "__main__":
#     # Create an instance of VideoProcessor
#     processor = VideoProcessor()

#     # Example direct mp4 URL
#     # direct_mp4_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

#     try:
#         # Process direct MP4 link
#         # print("\nPROCESSING DIRECT MP4:")
#         # print("-" * 50)
#         # direct_audio_path = processor.process_video(direct_mp4_url)
#         # print(f"Direct MP4 audio saved to: {direct_audio_path}")

#         # YouTube URL
#         youtube_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" (first YouTube video)
#         print("\nPROCESSING YOUTUBE VIDEO:")
#         print("-" * 50)
#         youtube_audio_path = processor.process_video(youtube_url)
#         print(f"YouTube audio saved to: {youtube_audio_path}")

#         print("\nVideo processing completed successfully!")
#     except Exception as e:
#         print(f"Error occurred: {e}")

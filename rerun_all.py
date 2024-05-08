import subprocess
from pathlib import Path


def main():
    pwd = Path(__file__).parent
    all_videos = pwd.glob("*.mkv")
    for video in all_videos:
        PYTHON = "python3.12"
        SCRIPT = "benchmark.py"
        args = [
            PYTHON,
            SCRIPT,
            video,
            "--scale",
            "0.18",
            "--video-output",
            "auto",
        ]
        subprocess.run(args, check=True)

if __name__ == "__main__":
    main()

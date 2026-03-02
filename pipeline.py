#!/usr/bin/env python3
"""
Subtitle generation pipeline.
Transcribes video/audio files using faster-whisper and outputs SRT subtitles.

Usage:
    python pipeline.py samples/Ilia\ Topuria\ Entrevista.mp4 --language es
    python pipeline.py path/to/video.mp4 -l es -o my_subtitles.srt
"""

import argparse
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp as _format_timestamp

OUTPUT_DIR = Path(__file__).parent / "output"
MAX_CHARS_PER_LINE = 42


def format_srt_timestamp(seconds: float) -> str:
    """SRT format: HH:MM:SS,mmm (comma as decimal marker)."""
    return _format_timestamp(seconds, always_include_hours=True, decimal_marker=",")


def wrap_text(
    text: str, 
    max_chars: int = MAX_CHARS_PER_LINE) -> list[str]:
    """Split text into lines of roughly max_chars for readability."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if not current_line:
            current_line.append(word)
            continue

        test_line = " ".join(current_line + [word])
        if len(test_line) <= max_chars:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines if lines else [""]


def segments_to_srt(
    segments, 
    word_timestamps: bool = False) -> str:
    """Convert transcription segments to SRT format."""
    srt_parts = []
    for i, segment in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(segment.start)
        end_ts = format_srt_timestamp(segment.end)
        text = segment.text.strip()
        if not text:
            continue

        lines = wrap_text(text)
        srt_parts.append(f"{i}\n{start_ts} --> {end_ts}\n" + "\n".join(lines) + "\n")

    return "\n".join(srt_parts)


def transcribe(
    input_path: Path,
    language: str | None = None,
    model_size: str = "small",
    vad_filter: bool = True,
    log_progress: bool = True,
):
    """Transcribe audio/video and return segments."""
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(input_path),
        language=language,
        beam_size=5,
        vad_filter=vad_filter,
        log_progress=log_progress,
    )
    segments = list(segments)
    return segments, info


def main():
    parser = argparse.ArgumentParser(
        description="Generate subtitles from video/audio using faster-whisper."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to video or audio file",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language code (e.g. es, en). Auto-detect if not specified.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output SRT path. Default: output/<input_name>.srt",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="small",
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar during transcription",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.output is not None:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = OUTPUT_DIR / output_path
    else:
        output_path = OUTPUT_DIR / f"{input_path.stem}.srt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Transcribing: {input_path}")
    print(f"Model: {args.model}, Language: {args.language or 'auto'}")
    segments, info = transcribe(
        input_path,
        language=args.language,
        model_size=args.model,
        log_progress=not args.no_progress,
    )

    print(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
    print(f"Segments: {len(segments)}")

    srt_content = segments_to_srt(segments)
    output_path.write_text(srt_content, encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

"""
Audio Rendering Pipeline — GAPT2026 Project 3: Sub-task 2
==========================================================
Usage:
    python audio_renderer.py --script adventure.json dogs.json magicians.json --api_key YOUR_KEY
"""

import argparse
import json
import re
import time
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings


# ── Voice map (confirmed available on this account) ────────────────────────────
#   Roger    CwhRBWXzGAHq8TQ4Fs17  male,   laid-back, casual
#   Sarah    EXAVITQu4vr4xnSDxMaL  female, mature, reassuring
#   Laura    FGY2WhTYpPnrIDTdsKH5  female, enthusiast, quirky
#   Charlie  IKne3meq5aSn9XLyUdCD  male,   deep, confident, energetic
#   George   JBFqnCBsd6RmkjVDRZzb  male,   warm, captivating
#   Callum   N2lVS1w4EtoT3dr4eOWO  male,   husky, trickster
#   Harry    SOYHLrjzK2X1ezoPC6cr  male,   fierce warrior
#   Liam     TX3LPaxmHKxFdv7VOQHJ  male,   energetic
#   Alice    Xb7hH8MSUJpSbSDYk0k2  female, clear, engaging
#   Jessica  cgSgspJ2msm6clMCkdW9  female, playful, bright, warm
#   Daniel   onwK4e9ZLuTAKqWW03F9  male,   steady broadcaster
#   Lily     pFZP5JQG7iQjIQuC4Bku  female, velvety actress
#   Brian    nPczCjzI2devNBz1zQrb  male,   deep, resonant

VOICE_MAP = {
    # ── adventure.json  ("A Chance Encounter") ──────────────────────────────
    "chance": {
        "kael":   {"voice_id": "TX3LPaxmHKxFdv7VOQHJ", "label": "Liam — energetic adventurer"},
        "lirien": {"voice_id": "cgSgspJ2msm6clMCkdW9", "label": "Jessica — playful elf"},
        "frost":  {"voice_id": "N2lVS1w4EtoT3dr4eOWO", "label": "Callum — husky wolf"},
    },

    # ── dogs.json  ("The Lonely Encounter") ─────────────────────────────────
    "lonely": {
        "duke":   {"voice_id": "CwhRBWXzGAHq8TQ4Fs17", "label": "Roger — gentle and lonely"},
        "luna":   {"voice_id": "EXAVITQu4vr4xnSDxMaL", "label": "Sarah — guarded and cautious"},
        "jasper": {"voice_id": "nPczCjzI2devNBz1zQrb", "label": "Brian — deep and gruff"},
    },

    # ── magicians.json  ("Mystic Duel") ─────────────────────────────────────
    "mystic": {
        "lyra": {"voice_id": "pFZP5JQG7iQjIQuC4Bku", "label": "Lily — cunning illusionist"},
        "kael": {"voice_id": "SOYHLrjzK2X1ezoPC6cr", "label": "Harry — fierce pyromancer"},
    },
}

FALLBACK_VOICE = {"voice_id": "CwhRBWXzGAHq8TQ4Fs17", "label": "Roger (fallback)"}

LINE_GAP_MS = 600
TARGET_DBFS = -18.0


def clean_for_tts(text: str) -> str:
    text = re.sub(r"\*[^*]+\*", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text or "..."


def resolve_voice(speaker: str, script_title: str) -> dict:
    title_lower = script_title.lower()
    speaker_lower = speaker.lower()
    for title_key, char_map in VOICE_MAP.items():
        if title_key in title_lower:
            if speaker_lower in char_map:
                return char_map[speaker_lower]
            print(f"  [WARN] Speaker '{speaker}' not in voice map.")
            return FALLBACK_VOICE
    print(f"  [WARN] Script '{script_title}' not recognised in voice map.")
    return FALLBACK_VOICE


def generate_tts(client: ElevenLabs, text: str, voice_id: str) -> bytes:
    audio_iter = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.45,
            similarity_boost=0.80,
            style=0.30,
            use_speaker_boost=True,
        ),
    )
    return b"".join(audio_iter)


def trim_silence(seg: AudioSegment, thresh=-40.0, pad=80) -> AudioSegment:
    start = detect_leading_silence(seg, silence_threshold=thresh)
    end = detect_leading_silence(seg.reverse(), silence_threshold=thresh)
    trimmed = seg[max(0, start - pad): len(seg) - max(0, end - pad)]
    return trimmed if len(trimmed) > 100 else seg


def normalise(seg: AudioSegment, target=TARGET_DBFS) -> AudioSegment:
    return seg.apply_gain(target - seg.dBFS)


def process_clip(raw_mp3: bytes) -> AudioSegment:
    import io
    seg = AudioSegment.from_file(io.BytesIO(raw_mp3), format="mp3")
    seg = trim_silence(seg)
    seg = normalise(seg)
    return seg


def run_pipeline(script_path: str, api_key: str, output_dir: str = "audio_output"):
    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    title = script.get("title", "scene")
    safe_title = title.replace(" ", "_")

    clips_dir = Path(output_dir) / "clips" / safe_title
    clips_dir.mkdir(parents=True, exist_ok=True)

    client = ElevenLabs(api_key=api_key)
    dialogue = script.get("dialogue_with_timing") or script.get("dialogue", [])

    timing_lines = []
    stitched = AudioSegment.empty()
    current_ms = 0

    print(f"\n{'='*62}")
    print(f"  {title}  |  {len(dialogue)} lines  |  → {output_dir}")
    print(f"{'='*62}")

    for i, line in enumerate(dialogue):
        speaker = line["speaker"]
        raw_text = line["text"]
        est_dur = line.get("estimated_duration_sec")
        clean_text = clean_for_tts(raw_text)
        voice = resolve_voice(speaker, title)

        print(f"\n  [{i+1:02d}/{len(dialogue):02d}] {speaker}  [{voice['label']}]")
        print(f"           \"{clean_text[:65]}{'...' if len(clean_text) > 65 else ''}\"")

        raw_mp3 = generate_tts(client, clean_text, voice["voice_id"])
        clip = process_clip(raw_mp3)

        fname = f"{i+1:02d}_{speaker.lower()}.mp3"
        clip.export(str(clips_dir / fname), format="mp3", bitrate="192k")

        actual_ms = len(clip)
        print(f"           → {fname}  {actual_ms/1000:.2f}s"
              + (f"  (est: {est_dur}s)" if est_dur else ""))

        timing_lines.append({
            "line_index": i + 1,
            "speaker": speaker,
            "original_text": raw_text,
            "tts_text": clean_text,
            "voice_id": voice["voice_id"],
            "voice_label": voice["label"],
            "start_ms": current_ms,
            "end_ms": current_ms + actual_ms,
            "duration_ms": actual_ms,
            "duration_sec": round(actual_ms / 1000, 3),
            "estimated_duration_sec": est_dur,
            "clip_file": str(clips_dir / fname),
        })

        stitched += clip
        current_ms += actual_ms

        if i < len(dialogue) - 1:
            stitched += AudioSegment.silent(duration=LINE_GAP_MS)
            current_ms += LINE_GAP_MS

        time.sleep(0.5)

    out_root = Path(output_dir)
    final_path = out_root / f"{safe_title}_final.mp3"
    stitched.export(str(final_path), format="mp3", bitrate="192k")

    metadata = {
        "title": title,
        "theme": script.get("theme"),
        "scene_summary": script.get("scene_summary"),
        "total_duration_ms": len(stitched),
        "total_duration_sec": round(len(stitched) / 1000, 3),
        "line_gap_ms": LINE_GAP_MS,
        "lines": timing_lines,
    }
    meta_path = out_root / f"{safe_title}_timing.json"
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✓ Final audio  → {final_path}  ({len(stitched)/1000:.1f}s)")
    print(f"  ✓ Timing JSON  → {meta_path}")
    print(f"{'='*62}\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAPT2026 Audio Renderer — Sub-task 2")
    parser.add_argument("--script", required=True, nargs="+")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--output", default="audio_output")
    args = parser.parse_args()

    for s in args.script:
        run_pipeline(script_path=s, api_key=args.api_key, output_dir=args.output)

import json
import re
from typing import List, Dict, Any, Optional

import ollama
from happytransformer import HappyTextToText, TTSettings


class StorySceneGenerator:
    def __init__(
        self,
        ollama_model: str = "llama3",
        use_happy_cleanup: bool = True,
        happy_model_name: str = "google/flan-t5-base",
        max_generation_retries: int = 3
    ):
        self.ollama_model = ollama_model
        self.use_happy_cleanup = use_happy_cleanup
        self.max_generation_retries = max_generation_retries

        self.happy = None
        self.happy_settings = None

        if self.use_happy_cleanup:
            try:
                self.happy = HappyTextToText("T5", happy_model_name)
                self.happy_settings = TTSettings(
                    max_length=512,
                    do_sample=False
                )
            except Exception as e:
                print(f"[Warning] Happy Transformer could not be loaded: {e}")
                print("[Warning] Continuing without Happy Transformer cleanup.")
                self.use_happy_cleanup = False


    def validate_input(
        self,
        story_sentence: str,
        theme: str,
        num_characters: int,
        duration_minutes: float
    ) -> None:
        if not isinstance(story_sentence, str) or not story_sentence.strip():
            raise ValueError("The story sentence cannot be empty.")

        if len(story_sentence.strip()) < 5:
            raise ValueError("The story sentence is too short.")

        if len(story_sentence.strip()) > 500:
            raise ValueError("The story sentence is too long. Keep it under 500 characters.")

        if not isinstance(theme, str) or not theme.strip():
            raise ValueError("The theme cannot be empty.")

        if len(theme.strip()) < 2:
            raise ValueError("The theme is too short.")

        if len(theme.strip()) > 100:
            raise ValueError("The theme is too long. Keep it under 100 characters.")

        if not isinstance(num_characters, int):
            raise ValueError("The number of characters must be an integer.")

        if num_characters < 1 or num_characters > 5:
            raise ValueError("The number of characters must be between 1 and 5.")

        if not isinstance(duration_minutes, (int, float)):
            raise ValueError("The duration must be a number.")

        if duration_minutes <= 0 or duration_minutes > 5:
            raise ValueError("The duration must be greater than 0 and at most 5 minutes.")

    def estimate_num_lines(self, duration_minutes: float) -> int:
        duration_seconds = duration_minutes * 60
        approx_lines = int(duration_seconds / 4)
        return max(4, min(approx_lines, 70))

    def estimate_min_max_lines(self, duration_minutes: float) -> Dict[str, int]:
        base = self.estimate_num_lines(duration_minutes)
        return {
            "min_lines": max(4, int(base * 0.8)),
            "target_lines": base,
            "max_lines": min(80, int(base * 1.2) + 1)
        }

    def build_prompt(
        self,
        story_sentence: str,
        theme: str,
        num_characters: int,
        duration_minutes: float
    ) -> str:
        line_info = self.estimate_min_max_lines(duration_minutes)

        return f"""
You are generating one original audiovisual micro-scene.

The output must strongly reflect ALL user inputs:
1. Story idea
2. Theme
3. Number of characters
4. Target duration

User inputs:
- Story idea: {story_sentence}
- Theme: {theme}
- Number of characters: {num_characters}
- Target duration: {duration_minutes} minutes

Strict requirements:
- Create exactly {num_characters} characters.
- Use only those characters in the dialogue.
- The scene must clearly reflect the story idea.
- The mood, language, and atmosphere must clearly reflect the theme.
- The amount of dialogue must reflect the duration.
- Aim for about {line_info["target_lines"]} dialogue lines.
- Keep the dialogue between {line_info["min_lines"]} and {line_info["max_lines"]} lines.
- Give each character a distinct role and personality.
- Make different characters speak differently.
- Keep the scene focused on one moment from a larger story.
- Dialogue should be suitable for later text-to-speech.
- No narration inside dialogue lines.
- No extra commentary outside JSON.

Return ONLY valid JSON with this exact structure:
{{
  "title": "scene title",
  "theme": "{theme}",
  "duration_minutes": {duration_minutes},
  "scene_summary": "2-4 sentence summary of the scene",
  "characters": [
    {{
      "name": "character name",
      "role": "character role",
      "personality": "short personality description"
    }}
  ],
  "dialogue": [
    {{
      "speaker": "character name",
      "text": "dialogue line"
    }}
  ]
}}
""".strip()

    def build_repair_prompt(
        self,
        broken_output: str,
        story_sentence: str,
        theme: str,
        num_characters: int,
        duration_minutes: float
    ) -> str:
        return f"""
The following model output is invalid or badly formatted.

Your task:
- repair it into valid JSON only
- preserve the original story content as much as possible
- ensure it follows the required structure exactly
- ensure there are exactly {num_characters} characters
- ensure the theme is "{theme}"
- ensure the duration_minutes field is {duration_minutes}
- ensure only listed characters appear in dialogue

Story idea: {story_sentence}
Theme: {theme}
Number of characters: {num_characters}
Duration: {duration_minutes}

Broken output:
{broken_output}

Return ONLY valid JSON:
{{
  "title": "scene title",
  "theme": "{theme}",
  "duration_minutes": {duration_minutes},
  "scene_summary": "2-4 sentence summary",
  "characters": [
    {{
      "name": "character name",
      "role": "character role",
      "personality": "short personality description"
    }}
  ],
  "dialogue": [
    {{
      "speaker": "character name",
      "text": "dialogue line"
    }}
  ]
}}
""".strip()


    def generate_with_ollama(self, prompt: str, temperature: float = 0.85) -> str:
        response = ollama.chat(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": temperature,
                "top_p": 0.95
            },
            format="json"
        )
        return response["message"]["content"]

    def try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def cleanup_text_happy(self, text: str) -> str:
        if not self.use_happy_cleanup or not self.happy:
            return text

        prompt = (
            "Rewrite the following so it is clean, well-formed, and easier to parse as structured data. "
            "Do not add explanations.\n"
            f"{text}"
        )
        try:
            result = self.happy.generate_text(prompt, args=self.happy_settings)
            return result.text.strip()
        except Exception:
            return text

    def validate_scene_structure(
        self,
        data: Dict[str, Any],
        expected_num_characters: int,
        expected_theme: str,
        expected_duration: float
    ) -> List[str]:
        errors = []

        required_top_keys = [
            "title",
            "theme",
            "duration_minutes",
            "scene_summary",
            "characters",
            "dialogue"
        ]

        for key in required_top_keys:
            if key not in data:
                errors.append(f"Missing top-level key: {key}")

        if not isinstance(data.get("title", ""), str) or not data.get("title", "").strip():
            errors.append("Title must be a non-empty string.")

        if not isinstance(data.get("scene_summary", ""), str) or not data.get("scene_summary", "").strip():
            errors.append("Scene summary must be a non-empty string.")

        actual_theme = str(data.get("theme", "")).strip()
        if not actual_theme:
            errors.append("Theme must be a non-empty string.")
        elif actual_theme.lower() != expected_theme.strip().lower():
            errors.append(
                f"Generated theme '{actual_theme}' does not match requested theme '{expected_theme}'."
            )

        try:
            actual_duration = float(data.get("duration_minutes"))
            if round(actual_duration, 2) != round(expected_duration, 2):
                errors.append(
                    f"Generated duration_minutes '{actual_duration}' does not match requested duration '{expected_duration}'."
                )
        except Exception:
            errors.append("duration_minutes must be numeric.")

        characters = data.get("characters", [])
        dialogue = data.get("dialogue", [])

        if not isinstance(characters, list):
            errors.append("'characters' must be a list.")
            characters = []

        if not isinstance(dialogue, list):
            errors.append("'dialogue' must be a list.")
            dialogue = []

        if len(characters) != expected_num_characters:
            errors.append(
                f"Expected {expected_num_characters} characters, but got {len(characters)}."
            )

        if len(characters) > 5:
            errors.append("More than 5 characters were generated.")

        character_names = set()
        for idx, c in enumerate(characters):
            if not isinstance(c, dict):
                errors.append(f"Character {idx} is not an object.")
                continue

            for field in ["name", "role", "personality"]:
                value = str(c.get(field, "")).strip()
                if not value:
                    errors.append(f"Character {idx} missing '{field}'.")

            name = str(c.get("name", "")).strip()
            if name:
                if name.lower() in {n.lower() for n in character_names}:
                    errors.append(f"Duplicate character name: {name}")
                character_names.add(name)

        if not dialogue:
            errors.append("Dialogue is empty.")

        for idx, line in enumerate(dialogue):
            if not isinstance(line, dict):
                errors.append(f"Dialogue line {idx} is not an object.")
                continue

            speaker = str(line.get("speaker", "")).strip()
            text = str(line.get("text", "")).strip()

            if not speaker:
                errors.append(f"Dialogue line {idx} missing speaker.")
            elif speaker not in character_names:
                errors.append(f"Dialogue line {idx} uses unknown speaker '{speaker}'.")

            if not text:
                errors.append(f"Dialogue line {idx} is empty.")

        return errors

    def scene_inputs_are_reflected(
        self,
        scene: Dict[str, Any],
        story_sentence: str,
        theme: str,
        num_characters: int,
        duration_minutes: float
    ) -> Dict[str, Any]:
        """
        A simple report that shows the user inputs affected the output.
        """
        dialogue = scene.get("dialogue", [])
        characters = scene.get("characters", [])
        summary = str(scene.get("scene_summary", "")).lower()
        title = str(scene.get("title", "")).lower()
        full_dialogue_text = " ".join(
            str(line.get("text", "")) for line in dialogue
        ).lower()

        combined_text = f"{title} {summary} {full_dialogue_text}"

        story_keywords = self.extract_keywords(story_sentence)
        theme_keywords = self.extract_keywords(theme)

        matched_story_keywords = [
            word for word in story_keywords if word.lower() in combined_text
        ]
        matched_theme_keywords = [
            word for word in theme_keywords if word.lower() in combined_text
        ]

        return {
            "requested_story_sentence": story_sentence,
            "requested_theme": theme,
            "requested_num_characters": num_characters,
            "requested_duration_minutes": duration_minutes,
            "actual_num_characters": len(characters),
            "actual_dialogue_lines": len(dialogue),
            "story_keyword_matches": matched_story_keywords,
            "theme_keyword_matches": matched_theme_keywords,
            "story_input_effect_detected": len(matched_story_keywords) > 0,
            "theme_input_effect_detected": len(matched_theme_keywords) > 0,
            "character_count_effect_detected": len(characters) == num_characters,
            "duration_effect_detected": self.dialogue_length_matches_duration(
                len(dialogue), duration_minutes
            )
        }

    def extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r"[A-Za-z0-9']+", text.lower())
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "to", "of", "and", "in",
            "on", "for", "with", "at", "by", "from", "that", "this", "it", "as",
            "he", "she", "they", "them", "his", "her", "their", "be", "into",
            "about", "after", "before", "over", "under"
        }
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        return list(dict.fromkeys(keywords))[:10]

    def dialogue_length_matches_duration(self, num_lines: int, duration_minutes: float) -> bool:
        bounds = self.estimate_min_max_lines(duration_minutes)
        return bounds["min_lines"] <= num_lines <= bounds["max_lines"]

    def estimate_dialogue_timing(
        self,
        dialogue: List[Dict[str, str]],
        words_per_second: float = 2.5
    ) -> List[Dict[str, Any]]:
        enriched = []
        for line in dialogue:
            text = str(line.get("text", "")).strip()
            word_count = len(text.split())
            estimated_duration = round(max(1.0, word_count / words_per_second), 2)

            enriched.append({
                "speaker": str(line.get("speaker", "")).strip(),
                "text": text,
                "word_count": word_count,
                "estimated_duration_sec": estimated_duration
            })
        return enriched

    def normalize_scene_json(
        self,
        data: Dict[str, Any],
        theme: str,
        duration_minutes: float
    ) -> Dict[str, Any]:
        normalized = {
            "title": str(data.get("title", "")).strip(),
            "theme": str(data.get("theme", theme)).strip(),
            "duration_minutes": float(data.get("duration_minutes", duration_minutes)),
            "scene_summary": str(data.get("scene_summary", "")).strip(),
            "characters": [],
            "dialogue": []
        }

        for c in data.get("characters", []):
            if isinstance(c, dict):
                normalized["characters"].append({
                    "name": str(c.get("name", "")).strip(),
                    "role": str(c.get("role", "")).strip(),
                    "personality": str(c.get("personality", "")).strip()
                })

        for line in data.get("dialogue", []):
            if isinstance(line, dict):
                normalized["dialogue"].append({
                    "speaker": str(line.get("speaker", "")).strip(),
                    "text": str(line.get("text", "")).strip()
                })

        return normalized

    def generate_scene(
        self,
        story_sentence: str,
        theme: str,
        num_characters: int,
        duration_minutes: float
    ) -> Dict[str, Any]:
        self.validate_input(
            story_sentence=story_sentence,
            theme=theme,
            num_characters=num_characters,
            duration_minutes=duration_minutes
        )

        prompt = self.build_prompt(
            story_sentence=story_sentence,
            theme=theme,
            num_characters=num_characters,
            duration_minutes=duration_minutes
        )

        last_raw_output = ""
        last_errors = []

        for attempt in range(1, self.max_generation_retries + 1):
            try:
                raw = self.generate_with_ollama(
                    prompt=prompt,
                    temperature=max(0.55, 0.9 - (attempt - 1) * 0.15)
                )
                last_raw_output = raw

                parsed = self.try_parse_json(raw)

                if parsed is None and self.use_happy_cleanup:
                    cleaned = self.cleanup_text_happy(raw)
                    parsed = self.try_parse_json(cleaned)

                if parsed is None:
                    repair_prompt = self.build_repair_prompt(
                        broken_output=raw,
                        story_sentence=story_sentence,
                        theme=theme,
                        num_characters=num_characters,
                        duration_minutes=duration_minutes
                    )
                    repaired_raw = self.generate_with_ollama(repair_prompt, temperature=0.3)
                    parsed = self.try_parse_json(repaired_raw)
                    if repaired_raw:
                        last_raw_output = repaired_raw

                if parsed is None:
                    last_errors = ["Failed to parse model output as JSON."]
                    continue

                normalized = self.normalize_scene_json(
                    data=parsed,
                    theme=theme,
                    duration_minutes=duration_minutes
                )

                errors = self.validate_scene_structure(
                    data=normalized,
                    expected_num_characters=num_characters,
                    expected_theme=theme,
                    expected_duration=duration_minutes
                )

                if errors:
                    last_errors = errors
                    continue

                normalized["dialogue_with_timing"] = self.estimate_dialogue_timing(
                    normalized.get("dialogue", [])
                )

                total_estimated_sec = round(
                    sum(line["estimated_duration_sec"] for line in normalized["dialogue_with_timing"]),
                    2
                )

                normalized["target_duration_sec"] = round(duration_minutes * 60, 2)
                normalized["total_estimated_dialogue_sec"] = total_estimated_sec
                normalized["estimated_line_target"] = self.estimate_num_lines(duration_minutes)
                normalized["input_effect_report"] = self.scene_inputs_are_reflected(
                    scene=normalized,
                    story_sentence=story_sentence,
                    theme=theme,
                    num_characters=num_characters,
                    duration_minutes=duration_minutes
                )
                normalized["validation_errors"] = []
                normalized["generation_attempt"] = attempt

                return normalized

            except Exception as e:
                last_errors = [f"Generation attempt {attempt} failed: {e}"]

        raise ValueError(
            "Generation failed after multiple attempts.\n"
            f"Last errors: {last_errors}\n"
            f"Last raw output: {last_raw_output}"
        )


def print_scene(scene: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"TITLE: {scene.get('title', 'Untitled')}")
    print(f"THEME: {scene.get('theme', '')}")
    print(f"TARGET DURATION: {scene.get('duration_minutes', '')} minutes")
    print(f"CHARACTERS REQUESTED: {len(scene.get('characters', []))}")
    print("=" * 70)

    print("\nSCENE SUMMARY:")
    print(scene.get("scene_summary", ""))

    print("\nCHARACTERS:")
    for c in scene.get("characters", []):
        print(
            f"- {c.get('name', 'Unknown')} | "
            f"Role: {c.get('role', '')} | "
            f"Personality: {c.get('personality', '')}"
        )

    print("\nDIALOGUE:")
    for i, line in enumerate(scene.get("dialogue_with_timing", []), start=1):
        print(
            f"{i:02d}. {line.get('speaker', 'Unknown')}: {line.get('text', '')} "
            f"[{line.get('word_count', 0)} words | ~{line.get('estimated_duration_sec', 0)} sec]"
        )

    print("\nTIMING REPORT:")
    print(f"- Target duration: {scene.get('target_duration_sec', 0)} sec")
    print(f"- Estimated dialogue duration: {scene.get('total_estimated_dialogue_sec', 0)} sec")
    print(f"- Estimated line target: {scene.get('estimated_line_target', 0)}")
    print(f"- Actual dialogue lines: {len(scene.get('dialogue', []))}")

    report = scene.get("input_effect_report", {})
    print("\nINPUT EFFECT REPORT:")
    print(f"- Story input affected output: {report.get('story_input_effect_detected', False)}")
    print(f"- Theme affected output: {report.get('theme_input_effect_detected', False)}")
    print(f"- Character count matched: {report.get('character_count_effect_detected', False)}")
    print(f"- Duration matched dialogue size: {report.get('duration_effect_detected', False)}")
    print(f"- Story keyword matches: {report.get('story_keyword_matches', [])}")
    print(f"- Theme keyword matches: {report.get('theme_keyword_matches', [])}")

    if scene.get("validation_errors"):
        print("\nVALIDATION WARNINGS:")
        for err in scene["validation_errors"]:
            print(f"- {err}")
    else:
        print("\nVALIDATION: OK")

    print(f"\nGeneration succeeded on attempt: {scene.get('generation_attempt', 1)}")
    print("=" * 70 + "\n")


def save_scene_to_json(scene: Dict[str, Any], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2, ensure_ascii=False)


def main():
    print("=== Story & Dialogue Generator (Ollama + Happy Transformer) ===")

    story_sentence = input("Enter the overall story sentence: ").strip()
    theme = input("Enter the theme: ").strip()

    try:
        num_characters = int(input("Enter number of characters (1-5): ").strip())
    except ValueError:
        print("Invalid number. Please enter an integer between 1 and 5.")
        return

    try:
        duration_minutes = float(input("Enter video duration (greater than 0 and up to 5 minutes): ").strip())
    except ValueError:
        print("Invalid duration. Please enter a number.")
        return

    use_happy_input = input("Use Happy Transformer cleanup? (y/n): ").strip().lower()
    use_happy_cleanup = use_happy_input == "y"

    generator = StorySceneGenerator(
        ollama_model="llama3",
        use_happy_cleanup=use_happy_cleanup,
        happy_model_name="google/flan-t5-base",
        max_generation_retries=3
    )

    try:
        scene = generator.generate_scene(
            story_sentence=story_sentence,
            theme=theme,
            num_characters=num_characters,
            duration_minutes=duration_minutes
        )

        print_scene(scene)

        save = input("Save output to JSON file? (y/n): ").strip().lower()
        if save == "y":
            filename = input("Enter filename (e.g. scene.json): ").strip()
            if not filename:
                filename = "scene.json"
            save_scene_to_json(scene, filename)
            print(f"Saved to {filename}")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
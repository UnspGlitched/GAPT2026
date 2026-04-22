import json
import re
import threading
from typing import List, Dict, Any, Optional

import ollama
from happytransformer import HappyTextToText, TTSettings
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


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
        lower_names = set()

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
                if name.lower() in lower_names:
                    errors.append(f"Duplicate character name: {name}")
                character_names.add(name)
                lower_names.add(name.lower())

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

    def scene_inputs_are_reflected(
        self,
        scene: Dict[str, Any],
        story_sentence: str,
        theme: str,
        num_characters: int,
        duration_minutes: float
    ) -> Dict[str, Any]:
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


class StorySceneGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Story & Dialogue Generator")
        self.root.geometry("1100x800")
        self.root.minsize(950, 700)

        self.generated_scene = None

        self.story_var = tk.StringVar()
        self.theme_var = tk.StringVar()
        self.characters_var = tk.StringVar(value="2")
        self.duration_var = tk.StringVar(value="1")
        self.cleanup_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="Scene Inputs", padding=12)
        input_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(input_frame, text="Story sentence:").grid(row=0, column=0, sticky="nw", padx=(0, 8), pady=6)
        self.story_text = tk.Text(input_frame, height=4, wrap="word")
        self.story_text.grid(row=0, column=1, sticky="ew", pady=6)

        ttk.Label(input_frame, text="Theme:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        self.theme_entry = ttk.Entry(input_frame, textvariable=self.theme_var)
        self.theme_entry.grid(row=1, column=1, sticky="ew", pady=6)

        options_frame = ttk.Frame(input_frame)
        options_frame.grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(options_frame, text="Characters (1-5):").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.characters_spin = ttk.Spinbox(options_frame, from_=1, to=5, textvariable=self.characters_var, width=8)
        self.characters_spin.grid(row=0, column=1, sticky="w", padx=(0, 20))

        ttk.Label(options_frame, text="Duration (0-5 min):").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.duration_entry = ttk.Entry(options_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.grid(row=0, column=3, sticky="w", padx=(0, 20))

        self.cleanup_check = ttk.Checkbutton(
            options_frame,
            text="Use Happy Transformer cleanup",
            variable=self.cleanup_var
        )
        self.cleanup_check.grid(row=0, column=4, sticky="w")

        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=1, sticky="w", pady=(10, 0))

        self.generate_button = ttk.Button(button_frame, text="Generate Scene", command=self.start_generation)
        self.generate_button.pack(side="left", padx=(0, 10))

        self.save_button = ttk.Button(button_frame, text="Save JSON", command=self.save_json, state="disabled")
        self.save_button.pack(side="left", padx=(0, 10))

        self.clear_button = ttk.Button(button_frame, text="Clear Output", command=self.clear_output)
        self.clear_button.pack(side="left")

        input_frame.columnconfigure(1, weight=1)

        output_frame = ttk.LabelFrame(main_frame, text="Generated Output", padding=12)
        output_frame.pack(fill="both", expand=True)

        self.output_text = tk.Text(output_frame, wrap="word", font=("Consolas", 10))
        self.output_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=scrollbar.set)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")

    def clear_output(self):
        self.output_text.delete("1.0", tk.END)
        self.generated_scene = None
        self.save_button.config(state="disabled")
        self.status_var.set("Output cleared.")

    def save_json(self):
        if not self.generated_scene:
            messagebox.showwarning("No Data", "There is no generated scene to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.generated_scene, f, indent=2, ensure_ascii=False)
            self.status_var.set(f"Saved JSON to: {filepath}")
            messagebox.showinfo("Saved", "Scene JSON saved successfully.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file:\n{e}")

    def start_generation(self):
        story_sentence = self.story_text.get("1.0", tk.END).strip()
        theme = self.theme_var.get().strip()
        num_characters_text = self.characters_var.get().strip()
        duration_text = self.duration_var.get().strip()
        use_cleanup = self.cleanup_var.get()

        try:
            num_characters = int(num_characters_text)
        except ValueError:
            messagebox.showerror("Input Error", "Number of characters must be an integer.")
            return

        try:
            duration_minutes = float(duration_text)
        except ValueError:
            messagebox.showerror("Input Error", "Duration must be a number.")
            return

        self.generate_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.status_var.set("Generating scene... Please wait.")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Generating scene...\n")

        thread = threading.Thread(
            target=self._run_generation,
            args=(story_sentence, theme, num_characters, duration_minutes, use_cleanup),
            daemon=True
        )
        thread.start()

    def _run_generation(
        self,
        story_sentence: str,
        theme: str,
        num_characters: int,
        duration_minutes: float,
        use_cleanup: bool
    ):
        try:
            generator = StorySceneGenerator(
                ollama_model="llama3",
                use_happy_cleanup=use_cleanup,
                happy_model_name="google/flan-t5-base",
                max_generation_retries=3
            )

            scene = generator.generate_scene(
                story_sentence=story_sentence,
                theme=theme,
                num_characters=num_characters,
                duration_minutes=duration_minutes
            )

            formatted_output = self.format_scene_for_display(scene)

            self.root.after(0, self._on_generation_success, scene, formatted_output)

        except Exception as e:
            self.root.after(0, self._on_generation_error, str(e))

    def _on_generation_success(self, scene: Dict[str, Any], formatted_output: str):
        self.generated_scene = scene
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, formatted_output)
        self.generate_button.config(state="normal")
        self.save_button.config(state="normal")
        self.status_var.set("Scene generated successfully.")

    def _on_generation_error(self, error_message: str):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Error:\n{error_message}")
        self.generate_button.config(state="normal")
        self.save_button.config(state="disabled")
        self.status_var.set("Generation failed.")
        messagebox.showerror("Generation Error", error_message)

    def format_scene_for_display(self, scene: Dict[str, Any]) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append(f"TITLE: {scene.get('title', 'Untitled')}")
        lines.append(f"THEME: {scene.get('theme', '')}")
        lines.append(f"TARGET DURATION: {scene.get('duration_minutes', '')} minutes")
        lines.append(f"GENERATION ATTEMPT: {scene.get('generation_attempt', '')}")
        lines.append("=" * 80)
        lines.append("")

        lines.append("SCENE SUMMARY:")
        lines.append(scene.get("scene_summary", ""))
        lines.append("")

        lines.append("CHARACTERS:")
        for c in scene.get("characters", []):
            lines.append(
                f"- {c.get('name', 'Unknown')} | "
                f"Role: {c.get('role', '')} | "
                f"Personality: {c.get('personality', '')}"
            )
        lines.append("")

        lines.append("DIALOGUE:")
        for i, line in enumerate(scene.get("dialogue_with_timing", []), start=1):
            lines.append(
                f"{i:02d}. {line.get('speaker', 'Unknown')}: {line.get('text', '')} "
                f"[{line.get('word_count', 0)} words | ~{line.get('estimated_duration_sec', 0)} sec]"
            )
        lines.append("")

        lines.append("TIMING REPORT:")
        lines.append(f"- Target duration: {scene.get('target_duration_sec', 0)} sec")
        lines.append(f"- Estimated dialogue duration: {scene.get('total_estimated_dialogue_sec', 0)} sec")
        lines.append(f"- Estimated line target: {scene.get('estimated_line_target', 0)}")
        lines.append(f"- Actual dialogue lines: {len(scene.get('dialogue', []))}")
        lines.append("")

        report = scene.get("input_effect_report", {})
        lines.append("INPUT EFFECT REPORT:")
        lines.append(f"- Story input affected output: {report.get('story_input_effect_detected', False)}")
        lines.append(f"- Theme affected output: {report.get('theme_input_effect_detected', False)}")
        lines.append(f"- Character count matched: {report.get('character_count_effect_detected', False)}")
        lines.append(f"- Duration matched dialogue size: {report.get('duration_effect_detected', False)}")
        lines.append(f"- Story keyword matches: {report.get('story_keyword_matches', [])}")
        lines.append(f"- Theme keyword matches: {report.get('theme_keyword_matches', [])}")
        lines.append("")

        validation_errors = scene.get("validation_errors", [])
        if validation_errors:
            lines.append("VALIDATION WARNINGS:")
            for err in validation_errors:
                lines.append(f"- {err}")
        else:
            lines.append("VALIDATION: OK")

        lines.append("")
        lines.append("RAW JSON PREVIEW:")
        lines.append(json.dumps(scene, indent=2, ensure_ascii=False))

        return "\n".join(lines)


def main():
    root = tk.Tk()
    app = StorySceneGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
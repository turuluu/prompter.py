"""
A Textual TUI to compare prompt wordings side-by-side with deterministic generations.

Key combos:
  1e/2e/3e = edit prompt in that column
  1d/2d/3d = mark worst & drop that column (increments stage)
  1g/2g/3g = generate alternative into that (empty) column (rewrites dropped prompt)
  r = run all (sequential, streaming)
  n = manually add new prompt into an empty slot
  m = change globals (model/temp/seed/num_ctx/num_predict etc.)
  s = save run
  o = open run JSON
  q = quit

Notes:
- Defaults: model='gpt-oss', deterministic temperature=0, fixed seed per run.
- Uses direct Ollama /api/generate for deterministic control; LlamaIndex used only for "generate alternative".
- No auto-pull: you manage models in Ollama.
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rich.pretty import Pretty
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll, HorizontalGroup
from textual.widgets import Static, RichLog, Button, TextArea, Label
from textual.widget import Widget
from textual.message import Message

SAVE_DIR = Path(os.path.expanduser("~/.prompt-test-saves"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Data
# -------------------------------

@dataclass
class GenStats:
    total_duration_ms: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration_ms: Optional[int] = None


@dataclass
class OutputRecord:
    timestamp: str
    text: str
    stats: GenStats = field(default_factory=GenStats)


@dataclass
class PromptSlot:
    slot_id: int
    text: str = ""
    active: bool = True  # false when dropped or empty
    empty: bool = True  # true if slot has no prompt yet
    dropped: bool = False
    born_stage: int = 0
    has_survived: bool = False  # flips true once it survives one drop after birth
    final_score: Optional[float] = None
    last_output: str = ""
    history: List[OutputRecord] = field(default_factory=list)
    last_dropped_text: Optional[str] = None  # remembers what was dropped (for generator)

    def current_score(self, stage: int) -> float:
        # Scoring rule:
        # - Global "stage" increments after each drop action.
        # - A prompt that has survived at least one drop while alive shows score = 1 / stage.
        # - A newly added prompt that hasn't yet survived a drop displays 1.0.
        # - When dropped at stage N, its final_score is 1 / N.
        if self.dropped and self.final_score is not None:
            return self.final_score
        if not self.empty and self.has_survived and stage > 0:
            return 1.0 / stage
        return 1.0


@dataclass
class RunSettings:
    model: str = 'gemma3:270m'
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    num_ctx: int = 2048  # default context length
    num_predict: int = 1024  # default max tokens to generate per output
    seed: int = 42
    base_url: str = "http://localhost:11434"  # Ollama


@dataclass
class GeneratorSettings:
    # For "Generate alternative" only (non-deterministic, separate from evaluation runs)
    model: str = 'mistral-nemo'
    temperature: float = 0.7
    top_p: float = 0.95
    num_ctx: int = 8192
    num_predict: int = 512


@dataclass
class RunState:
    run_id: str
    created_at: str
    stage: int = 0  # increments after each drop
    settings: RunSettings = field(default_factory=RunSettings)
    gen_settings: GeneratorSettings = field(default_factory=GeneratorSettings)
    slots: List[PromptSlot] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "RunState":
        settings = RunSettings(**data["settings"])
        gens = GeneratorSettings(**data.get("gen_settings", {}))
        slots = []
        for s in data["slots"]:
            ps = PromptSlot(
                slot_id=s["slot_id"],
                text=s.get("text", ""),
                active=s.get("active", True),
                empty=s.get("empty", True),
                dropped=s.get("dropped", False),
                born_stage=s.get("born_stage", 0),
                has_survived=s.get("has_survived", False),
                final_score=s.get("final_score"),
                last_output=s.get("last_output", ""),
                history=[
                    OutputRecord(
                        timestamp=rec.get("timestamp", ""),
                        text=rec.get("text", ""),
                        stats=GenStats(
                            total_duration_ms=rec.get("stats", {}).get("total_duration_ms"),
                            eval_count=rec.get("stats", {}).get("eval_count"),
                            eval_duration_ms=rec.get("stats", {}).get("eval_duration_ms"),
                        ),
                    )
                    for rec in s.get("history", [])
                ],
                last_dropped_text=s.get("last_dropped_text"),
            )
            slots.append(ps)
        return RunState(
            run_id=data["run_id"],
            created_at=data["created_at"],
            stage=data.get("stage", 0),
            settings=settings,
            gen_settings=gens,
            slots=slots,
            note=data.get("note", ""),
        )


class Status(Message):
    def __init__(self, msg: str):
        super().__init__()
        now = datetime.now().strftime('%H:%M:%S')
        lines = (msg + '\n').splitlines()
        lines[0] = f'{lines[0]:<70}{now}'
        self.msg = '\n'.join(lines)


# -------------------------------
# UI widgets
# -------------------------------

class InlineEditor(Static):
    BINDINGS = [
        ("ctrl+s", "save", "Save"),
        ("escape", "cancel", "Cancel"),
    ]

    class Saved(Message):
        def __init__(self, sender: Widget, text: str) -> None:
            super().__init__()
            self.text = text

    class Cancelled(Message):
        pass

    def __init__(self, initial: str = '', title: str = '', hide_buttons: bool = False) -> None:
        super().__init__(classes="editor hidden")
        self._title = title
        self._initial = initial
        self._hide_buttons = hide_buttons
        self._area: TextArea | None = None

    def compose(self):
        if self._title:
            yield Label(self._title, classes="editor-title")
        self._area = TextArea()
        yield self._area
        if not self._hide_buttons:
            with HorizontalGroup():
                yield Button("save", classes='small-btn', id="save", variant="primary")
                yield Button("cancel", classes='small-btn', id="cancel")

    # ----- Callbacks

    def on_mount(self):
        self.set_text(self._initial)
        self.focus_input()

    def action_save(self):
        self.post_message(self.Saved(self, self._area.text if self._area else ""))

    def action_cancel(self):
        self.post_message(self.Cancelled())

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save":
            self.action_save()
        else:
            self.action_cancel()

    # ----- Explicit methods

    def edit(self, text: str):
        self.set_text(text)
        self.remove_class("hidden")
        self.focus_input()

    def set_text(self, text: str) -> None:
        if self._area:
            self._area.text = text

    def focus_input(self):
        if self._area:
            self._area.focus()


class Hints(Static):
    def __init__(self, state: RunState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label(
            "\nHints: \n• '1e/2e/3e' edit \n• '1d/2d/3d' drop \n• '1g/2g/3g' alt \n• r run \n• n new \n• m globals \n• s save \n• o open \n• q quit",
            classes="hints")


class Logs(Static):
    def __init__(self, state: RunState) -> None:
        super().__init__()
        self.state = state
        self.status_log: Optional[RichLog] = None

    def compose(self) -> ComposeResult:
        yield Label("\nStatus", classes="side-title")
        self.status_log = RichLog(id="status-log")
        yield self.status_log

    def post_status(self, msg: str) -> None:
        if self.status_log is not None:
            self.status_log.write(msg)


class SettingsPanel(Static):
    """Compact header showing globals & stage."""

    def __init__(self, state: RunState) -> None:
        super().__init__()
        self.state = state
        self.editor: InlineEditor | None = None
        self.panel: Static | None = None

    def compose(self) -> ComposeResult:
        # editor overlays on top of the panel when visible
        self.editor = InlineEditor(hide_buttons=True)
        yield self.editor

        self.panel = Static()
        yield self.panel

    # ----- Callbacks

    def on_mount(self):
        self.refresh_panel()

    def on_inline_editor_saved(self, msg: InlineEditor.Saved):
        try:
            text = (msg.text or "").strip()
            new_settings: RunSettings = RunSettings(**json.loads(text))
            self.state.settings = new_settings
            self.refresh_panel()
            self.post_message(Status('Settings changed'))
            self.close_editor()
        except Exception as e:
            self.post_message(Status('JSON syntax or schema error - changed settings discarded...'))

    def on_inline_editor_cancelled(self, _msg: InlineEditor.Cancelled):
        self.close_editor()

    # ----- Explicit methods

    def refresh_panel(self) -> None:
        """Recompute and update the header content."""
        s = self.state.settings
        self.panel.update(Pretty({
            "model": s.model,
            "temp": s.temperature,
            "seed": s.seed,
            "num_ctx": s.num_ctx,
            "num_predict": s.num_predict,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "repeat_penalty": s.repeat_penalty,
            "stage": self.state.stage,
        },
            expand_all=True))

    def open_editor(self):
        assert self.editor is not None
        settings = json.dumps(asdict(self.state.settings), indent=2)
        self.editor.edit(settings)

    def close_editor(self):
        if self.editor:
            self.editor.add_class("hidden")


class PromptPanel(Static):
    """One column: prompt text + output log."""

    class Edited(Message):
        def __init__(self, slot_id: int, new_text: str) -> None:
            self.slot_id = slot_id
            self.new_text = new_text
            super().__init__()

    def __init__(self, slot: PromptSlot) -> None:
        super().__init__()
        self.slot = slot
        self.prompt_label = Label("", id=f"prompt-label-{slot.slot_id}")
        self.prompt_text = RichLog(id=f"prompt-text-{slot.slot_id}", highlight=False)
        self.response = RichLog(id=f"output-log-{slot.slot_id}", highlight=False, wrap=True, min_width=40)
        self.editor: InlineEditor | None = None
        self.line = ''

    def compose(self) -> ComposeResult:
        # editor overlays on top of the panel when visible
        self.editor = InlineEditor(self.slot.text, title='Edit prompt (Ctrl+S to save, Esc to cancel)')
        yield self.editor
        yield Label(f"Prompt {self.slot.slot_id}", classes="title")
        yield self.prompt_text
        yield Label("Output", classes="title")
        yield self.response

    def open_editor(self):
        assert self.editor is not None
        # seed editor with current text and show it
        self.editor.edit(self.slot.text)

    def close_editor(self):
        if self.editor:
            self.editor.add_class("hidden")

    def on_inline_editor_saved(self, msg: InlineEditor.Saved):
        text = (msg.text or "").strip()
        self.slot.text = text
        self.slot.empty = (text == "")
        self.slot.active = not self.slot.empty
        self.refresh_prompt()
        self.close_editor()
        # tell siblings (side panel / header) to refresh
        self.post_message(self.Edited(self.slot.slot_id, text))

    def on_inline_editor_cancelled(self, _msg: InlineEditor.Cancelled):
        self.close_editor()

    def refresh_prompt(self) -> None:
        self.prompt_text.clear()
        if self.slot.empty:
            self.prompt_text.write("[empty slot]  (use 'n' to add or '{id}g' to generate)".format(id=self.slot.slot_id))
        else:
            self.prompt_text.write(self.slot.text)

    def clear_output(self) -> None:
        self.response.clear()

    def append_output(self, text: str) -> None:
        lines = text.splitlines()
        if len(lines) > 1:
            self.line += lines[0]
            self.response.write(self.line)
            self.line = lines[1]
        else:
            self.line += lines[0]

    def replace_output(self, text: str) -> None:
        self.response.clear()
        if text:
            self.response.write(text)


class SidePanel(VerticalScroll):
    """Shows scores and status for all prompts."""

    def __init__(self, state: RunState) -> None:
        super().__init__()
        self.state = state
        self.labels: Dict[int, Label] = {}

    def compose(self) -> ComposeResult:
        yield Label("Prompts & Scores", classes="side-title")
        for s in self.state.slots:
            lbl = Label("", id=f"score-{s.slot_id}")
            self.labels[s.slot_id] = lbl
            yield lbl

    def update_panel(self) -> None:

        for s in self.state.slots:
            status = "-" if s.empty else ("x" if s.dropped else "+")
            score = s.current_score(self.state.stage)
            text = f"[{s.slot_id}] {status}  score={score:.4f}"
            if not s.empty and not s.dropped:
                text += f"  born@{s.born_stage}  survived={s.has_survived}"
            self.labels[s.slot_id].update(text)


# -------------------------------
# App
# -------------------------------

class PromptEvalApp(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 4 2;
        grid-rows: 13 1fr;
        grid-columns: 22 1fr 1fr 1fr;
    }
    .title { content-align: left middle; padding: 0 1; }
    .side-title { padding: 0 1; }
    #side    { row-span: 1; border: round darkslategrey; }
    #header  { height:20; column-span: 1; border: round darkslategrey; }
    #hints   { padding: 0 1; column-span: 1; border: round darkslategrey; }
    #logs    { column-span: 2; border: round darkslategrey; }
    
    /* Make each column a stacking context and allow an overlay */
    #col1, #col2, #col3 { 
        position: relative; 
        border: round darkslategrey;
    }

    .editor {
        layer: overlay;        /* render above normal content */
        dock: top;
        height: auto;
        background: $boost;    /* subtle elevated background */
    }
    .small-btn {
        padding: 0 0;       /* less horizontal padding */
        height: auto;       /* shrink vertically */
        min-width: 6;       /* narrower minimum width */
        content-align: center middle;
    }
    .editor-title { padding: 0 1; }
    .hidden { display: none; }
    RichLog { height: 1fr; }
    #status-log { height: 10; }  /* optional: give status a compact fixed height */
    """

    BINDINGS = [
        Binding("r", "run_all", "Run"),
        Binding("n", "add_new", "New"),
        Binding("m", "edit_globals", "Globals"),
        Binding("s", "save_run", "Save"),
        Binding("o", "open_run", "Open"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = self._new_run_state()
        self.settings_panel = SettingsPanel(self.state)
        self.side_panel = SidePanel(self.state)
        self.hints_panel = Hints(self.state)
        self.logs_panel = Logs(self.state)
        self.panels: Dict[int, PromptPanel] = {}
        self.combo_buffer: Optional[int] = None
        self.combo_ts: float = 0.0
        self.client: Optional[httpx.AsyncClient] = None
        self.run_lock = asyncio.Lock()

    def _new_run_state(self) -> RunState:
        return RunState(
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            stage=0,
            settings=RunSettings(),
            gen_settings=GeneratorSettings(),
            slots=[PromptSlot(slot_id=i + 1, empty=True) for i in range(3)],
        )

    def compose(self) -> ComposeResult:
        # Order matters for grid placement.
        self.side_panel.id = "side"
        yield self.side_panel
        self.settings_panel.id = "header"
        yield self.settings_panel
        self.logs_panel.id = "logs"
        yield self.logs_panel
        self.hints_panel.id = "hints"
        yield self.hints_panel
        self.state.slots[0].text = 'What is the capital of France?'
        self.state.slots[0].empty = False
        for i in range(1, 4):
            panel = PromptPanel(self.state.slots[i - 1])
            panel.id = f"col{i}"
            panel.refresh_prompt()
            self.panels[i] = panel
            yield panel

    async def on_mount(self) -> None:
        # Side effects only (no layout API calls here on latest Textual).
        self.client = httpx.AsyncClient(timeout=30.0)
        self.side_panel.update_panel()
        self.side_panel.focus()

    async def on_unmount(self) -> None:
        if self.client:
            await self.client.aclose()

    def on_prompt_panel_edited(self, msg: PromptPanel.Edited) -> None:
        self.side_panel.update_panel()
        self.settings_panel.refresh_panel()

    # ----- Callbacks

    def on_status(self, msg: Status):
        self.logs_panel.post_status(msg.msg)

    async def on_key(self, event) -> None:
        key = event.key
        now = time.time()

        if key in ("1", "2", "3"):
            self.combo_buffer = int(key)
            self.combo_ts = now
            return

        if self.combo_buffer and (now - self.combo_ts) < 1.2:
            slot = self.combo_buffer
            self.combo_buffer = None

            if key == "e":
                await self._edit_prompt(slot)
                return
            if key == "d":
                await self._drop_slot(slot)
                return
            if key == "g":
                await self._generate_alternative(slot)
                return
            if key == "n":
                s = self.state.slots[slot - 1]
                if not s.empty:
                    await self._status(f"Slot {slot} not empty. Drop first with '{slot}d'.")
                    return
                await self._edit_prompt(slot)
                return
        # fall through to normal single-key bindings

    # -------------- Actions

    async def action_run_all(self) -> None:
        if self.run_lock.locked():
            await self._status("Already running…")
            return
        async with self.run_lock:
            # Sequential runs across active, non-empty slots (1->3)
            await self._status(f"Starting run… model={self.state.settings.model}")
            for i in range(1, 4):
                slot = self.state.slots[i - 1]
                if slot.empty or slot.dropped:
                    await self._status(f"Skip slot {i}: empty={slot.empty} dropped={slot.dropped}")
                    continue
                await self._status(f"Running slot {i}…")
                await self._run_one(i)
            await self._status("Run complete.")

    async def action_add_new(self) -> None:
        # find first empty slot
        idx = next((i for i, s in enumerate(self.state.slots) if s.empty), None)
        if idx is None:
            await self._status("No empty slot. Drop one first with 'Xd'.")
            return

        idx = next((i for i, s in enumerate(self.state.slots) if s.empty), None)
        if idx is None:
            await self._status("No empty slot. Drop one first with 'Xd'.")
            return
        slot = idx + 1
        self.run_worker(self._edit_prompt(slot), exclusive=True)

    async def action_edit_globals(self) -> None:
        self.settings_panel.open_editor()

    async def action_save_run(self) -> None:
        path = SAVE_DIR / f"run-{self.state.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_json(), f, indent=2)
        await self._status(f"Saved: {path}")

    async def action_open_run(self) -> None:
        path = await self._prompt_line("Open run path (JSON)", str(SAVE_DIR))
        if not path:
            return
        p = Path(path).expanduser()
        if not p.exists():
            await self._status("File not found.")
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.state = RunState.from_json(data)
            self.settings_panel.state = self.state
            self.side_panel.state = self.state
            self.hints_panel.state = self.state
            self.logs_panel.state = self.state
            for i in range(3):
                self.panels[i + 1].slot = self.state.slots[i]
            self.refresh_all()
        except Exception as e:
            await self._status(f"Open failed: {e}")

    async def action_quit(self) -> None:
        # Try to save; don’t block quitting if it fails.
        try:
            await self.action_save_run()
        except Exception as e:
            await self._status(f"Save failed: {e}")
        # Clean up background work & HTTP client, then exit the app.
        try:
            self.cancel_all_workers()
        except Exception:
            pass
        if getattr(self, "client", None) is not None:
            try:
                await self.client.aclose()
            except Exception:
                pass
        self.exit()

    # -------------- Helpers

    async def _status(self, msg: str) -> None:
        """Write status to the side panel and to a log file."""
        # On-screen
        try:
            self.post_message(Status(msg))
        except Exception:
            pass

        # File log
        try:
            with (SAVE_DIR / "app.log").open("a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass

    async def _edit_prompt(self, slot_id: int) -> None:
        panel = self.panels[slot_id]
        panel.open_editor()

    async def _drop_slot(self, slot_id: int) -> None:
        slot = self.state.slots[slot_id - 1]
        if slot.empty or slot.dropped:
            await self._status("Slot already empty/dropped.")
            return
        # Perform drop: increment stage, compute scores, mark survivors as survived
        # Stage increments AFTER the drop. The dropped prompt final_score = 1 / stage.
        self.state.stage += 1
        slot.dropped = True
        slot.active = False
        slot.last_dropped_text = slot.text
        slot.final_score = 1.0 / self.state.stage if self.state.stage > 0 else 1.0
        # mark others as having survived at least one drop
        for s in self.state.slots:
            if s is not slot and not s.empty and not s.dropped:
                s.has_survived = True
        # Clear prompt content to free the slot but retain last_dropped_text for generator.
        slot.text = ""
        slot.empty = True
        slot.last_output = ""
        self.panels[slot_id].slot = slot
        self.panels[slot_id].refresh_prompt()
        self.panels[slot_id].clear_output()
        self.side_panel.update_panel()
        self.settings_panel.refresh_panel()

    async def _generate_alternative(self, slot_id: int) -> None:
        slot = self.state.slots[slot_id - 1]
        await self._status(f"Generating alternative with {self.state.gen_settings.model} …")
        if not slot.empty:
            await self._drop_slot(slot_id)
        base_text = slot.last_dropped_text or ""
        if not base_text.strip():
            await self._status(f"Slot {slot_id} has no dropped prompt to rewrite. Drop here first with '{slot_id}d'.")
            return
        try:
            alt = await self._rewrite_prompt(base_text)
        except Exception as e:
            await self._status(f"Generate alternative failed: {e!s}")
            return
        # Insert new prompt
        slot.text = alt.strip()
        slot.empty = False
        slot.active = True
        slot.dropped = False
        slot.born_stage = self.state.stage
        slot.has_survived = False
        slot.last_output = ""
        slot.history.clear()
        self.panels[slot_id].slot = slot
        self.panels[slot_id].refresh_prompt()
        self.side_panel.update_panel()

    async def _run_one(self, slot_id: int) -> None:
        slot = self.state.slots[slot_id - 1]
        panel = self.panels[slot_id]
        panel.clear_output()
        slot.last_output = ""
        try:
            async for chunk in self._ollama_stream(slot.text):
                if chunk is None:
                    continue
                panel.append_output(chunk)
                slot.last_output += chunk
            panel.append_output('\n\n')
            # end-of-stream: record stats are attached via last meta
        except Exception as e:
            panel.append_output(f"\n[ERROR] {e}")
        # store history
        rec = OutputRecord(
            timestamp=datetime.now(UTC).isoformat(),
            text=slot.last_output,
            stats=GenStats(),  # filled by _ollama_stream end meta if available
        )
        slot.history.append(rec)

    async def _ollama_stream(self, prompt: str):
        """Stream generation from Ollama with locked params; yields text chunks."""
        s = self.state.settings
        url = f"{s.base_url.rstrip('/')}/api/generate"
        await self._status(f"POST {url} model={s.model}")
        payload = {
            "model": s.model,
            "prompt": prompt,
            "stream": True,
            "seed": s.seed,
            "options": {
                "temperature": s.temperature,
                "top_p": s.top_p,
                "top_k": s.top_k,
                "repeat_penalty": s.repeat_penalty,
                "presence_penalty": s.presence_penalty,
                "frequency_penalty": s.frequency_penalty,
                "num_ctx": min(max(512, s.num_ctx), 131072),
                "num_predict": min(max(1, s.num_predict), 131072),
            },
        }
        try:
            async with self.client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Some servers may send "data: {...}" style; strip prefix
                        if line.startswith("data:"):
                            try:
                                obj = json.loads(line[5:].strip())
                            except Exception:
                                continue
                        else:
                            continue
                    if "response" in obj and obj.get("done") is not True:
                        yield obj["response"]
                    if obj.get("done"):
                        # Final stats could be used if desired
                        break
        except httpx.HTTPError as e:
            await self._status(f"Ollama HTTP error: {e!s}")
            raise
        except Exception as e:
            await self._status(f"Ollama stream error: {e!s}")
            raise

    async def _rewrite_prompt(self, dropped_text: str) -> str:
        """Use LlamaIndex (Ollama LLM) to propose a revised alternative of dropped prompt only."""
        try:
            from llama_index.llms.ollama import Ollama  # type: ignore
        except Exception:
            # Fallback: ask Ollama directly with a non-deterministic call
            return await self._rewrite_prompt_fallback(dropped_text)

        await self._status('rewriting using llama_index...')
        g = self.state.gen_settings
        llm = Ollama(
            model=g.model,
            base_url=self.state.settings.base_url,
            temperature=g.temperature,
            request_timeout=30.0,
            additional_kwargs={
                "top_p": g.top_p,
                "num_ctx": g.num_ctx,
                "num_predict": g.num_predict,
            },
        )
        system = (
            'You are an automatic, technical copy-writer for LLM prompts.'
        )
        user = (
            "Rewrite the provided prompt into a clear, distinct alternative for the SAME task. "
            "Respond using ONLY the rewritten prompt."
            f"Reword this prompt improving its format for more specific LLM runs:\n{dropped_text}"
        )
        resp = await llm.acomplete(system_prompt=system, prompt=user)
        text = getattr(resp, "text", None) or str(resp)
        return text.strip()

    async def _rewrite_prompt_fallback(self, dropped_text: str) -> str:
        # Non-deterministic generator via direct Ollama; higher temp
        await self._status("Using fallback generator...")
        g = self.state.gen_settings
        s = self.state.settings
        url = f"{s.base_url.rstrip('/')}/api/generate"
        prompt = (
            "You rewrite prompts for A/B testing of wording only. "
            "Rewrite the provided prompt into a clear, distinct alternative for the SAME task. "
            "Do not add new capabilities, tool calls, or constraints. Keep it roughly same length.\n\n"
            f"Original prompt:\n---\n{dropped_text}\n---\nReturn ONLY the rewritten prompt."
        )
        payload = {
            "model": g.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": g.temperature,
                "top_p": g.top_p,
                "num_ctx": g.num_ctx,
                "num_predict": g.num_predict,
            },
        }
        try:
            resp = await self.client.post(url, json=payload, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            await self._status(f"Generator HTTP error: {e!s}")
            raise
        data = resp.json()
        return data.get("response", "").strip()


if __name__ == "__main__":
    try:
        app = PromptEvalApp()
        app.run()
    except KeyboardInterrupt:
        pass

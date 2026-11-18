"""Main window: project open/create, script editing, cut review, export."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from openclip.ai.beauty import BeautyConfig, beauty_available
from openclip.ai.broll_detection import BrollDetector, BrollOpportunity
from openclip.ai.broll_sourcing import BrollSource, BrollSourceResolver
from openclip.ai.hook_detection import Hook
from openclip.ai.reframing import DEFAULT_TARGET_ASPECT, tracking_from_default_face_analysis
from openclip.engine.captions import ANIMATIONS
from openclip.engine.enhance import PRESET_NAMES
from openclip.engine.ffmpeg import MediaInfo, probe
from openclip.engine.pipeline import available_transcribers
from openclip.engine.project import Project
from openclip.gui.preview import VideoPreviewWidget
from openclip.gui.reframe_preview import ReframePreviewWidget
from openclip.gui.workers import (
    AIPlanRequest,
    AIPlanWorker,
    AIRenderRequest,
    AIRenderWorker,
    AnalysisRequest,
    AnalysisWorker,
    RenderRequest,
    RenderWorker,
    ThreadedRunner,
)

_SCRIPT_HELP = (
    "Paste your lesson script. One block per sentence, separated by blank "
    "lines. Use 'primary text :: support text' for bilingual pairs.\n\n"
    "Example:\n"
    "Hello class :: 大家好\n"
    "\n"
    "Today we learn five words.\n"
    "今天我们学五个单词。"
)


class MainWindow(QMainWindow):
    media_loaded = Signal(MediaInfo)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Eterna OpenClip Studio")
        self.resize(1180, 760)

        self.project: Project | None = None
        self.media: MediaInfo | None = None
        self._project_path: Path | None = None
        self._analysis_runner = ThreadedRunner()
        self._render_runner = ThreadedRunner()
        self._ai_plan_runner = ThreadedRunner()
        self._ai_render_runner = ThreadedRunner()
        self._ai_plan: object | None = None  # AIDirectorPlan
        self._original_transcript: object | None = None  # parsed transcript JSON/text
        self._transcript_decisions: list[object] = []  # Decision list from the diff
        self._hooks: list[Hook] = []  # detected viral moments (Top Moments table)
        self._broll_opportunities: list[BrollOpportunity] = []  # detected B-roll opportunities
        self._broll_sources: list[BrollSource] = []  # resolved B-roll sources

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addLayout(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        splitter.addWidget(self._build_script_panel())
        splitter.addWidget(self._build_review_panel())
        splitter.addWidget(self._build_transcript_panel())
        splitter.addWidget(self._build_ai_panel())
        splitter.addWidget(self._build_moments_panel())
        splitter.addWidget(self._build_broll_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.setSizes([280, 300, 260, 280, 240, 220, 220])

        root.addWidget(self._build_player_bar())
        self.setStatusBar(QStatusBar())

    # ---------- construction helpers ----------

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.btn_new = QPushButton("New project…")
        self.btn_open = QPushButton("Open project…")
        self.btn_save = QPushButton("Save project")
        self.btn_analyze = QPushButton("Analyze")
        self.btn_export = QPushButton("Export MP4…")
        self.media_label = QLabel("No media loaded")
        for w in (
            self.btn_new,
            self.btn_open,
            self.btn_save,
            self.btn_analyze,
            self.btn_export,
        ):
            bar.addWidget(w)
        bar.addWidget(self.media_label, 1)
        self.btn_new.clicked.connect(self.new_project)
        self.btn_open.clicked.connect(self.open_project)
        self.btn_save.clicked.connect(self.save_project)
        self.btn_analyze.clicked.connect(self.analyze)
        self.btn_export.clicked.connect(self.export)
        self.btn_save.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_export.setEnabled(False)
        return bar

    def _build_script_panel(self) -> QWidget:
        box = QGroupBox("Lesson script (bilingual)")
        layout = QVBoxLayout(box)
        self.script_edit = QPlainTextEdit()
        self.script_edit.setPlaceholderText(_SCRIPT_HELP)
        layout.addWidget(self.script_edit)
        hint = QLabel("Blocks separated by blank lines; 'primary :: support' pairs.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _build_review_panel(self) -> QWidget:
        box = QGroupBox("Proposed cuts (review before export)")
        layout = QVBoxLayout(box)
        self.cut_table = QTableWidget(0, 5)
        self.cut_table.setHorizontalHeaderLabels(
            ["Keep?", "Kind", "Start (s)", "End (s)", "Reason"]
        )
        self.cut_table.horizontalHeader().setStretchLastSection(True)
        self.cut_table.cellDoubleClicked.connect(self._jump_to_cut)
        layout.addWidget(self.cut_table)
        note = QLabel("Uncheck a row to KEEP that span; checked rows are removed.")
        layout.addWidget(note)
        return box

    def _build_transcript_panel(self) -> QWidget:
        box = QGroupBox("Transcript editing (edit text → propose cuts)")
        layout = QVBoxLayout(box)
        self.transcript_edit = QPlainTextEdit()
        self.transcript_edit.setPlaceholderText(
            "Original transcript text. Load a timestamped transcript JSON, or edit "
            "the script-derived text here, then Apply Transcript Edits."
        )
        layout.addWidget(self.transcript_edit)
        btn_row = QHBoxLayout()
        self.btn_load_transcript = QPushButton("Load transcript…")
        self.btn_apply_transcript = QPushButton("Apply Transcript Edits")
        btn_row.addWidget(self.btn_load_transcript)
        btn_row.addWidget(self.btn_apply_transcript)
        layout.addLayout(btn_row)
        hint = QLabel(
            "Deleted blocks become proposed cuts, shown in the AI Director panel "
            "for review before rendering."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.btn_load_transcript.clicked.connect(self.load_transcript)
        self.btn_apply_transcript.clicked.connect(self.apply_transcript_edits)
        self.btn_load_transcript.setEnabled(False)
        self.btn_apply_transcript.setEnabled(False)
        return box

    def _build_ai_panel(self) -> QWidget:
        box = QGroupBox("AI Director proposals (review before render)")
        layout = QVBoxLayout(box)
        self.edit_ai_prompt = QLineEdit()
        self.edit_ai_prompt.setPlaceholderText(
            "Natural-language request, e.g. 'Make a short clip about the main topic'"
        )
        layout.addWidget(self.edit_ai_prompt)

        # BYOK: provider selector + secure (password-masked) API key field.
        # The key is passed in-memory to the worker only — never logged/saved.
        prov_row = QHBoxLayout()
        self.cmb_ai_provider = QComboBox()
        from openclip.ai.config import PROVIDER_NAMES

        for name in PROVIDER_NAMES:
            self.cmb_ai_provider.addItem(name)
        self.cmb_ai_provider.setCurrentText("local_mock")
        prov_row.addWidget(QLabel("Provider:"))
        prov_row.addWidget(self.cmb_ai_provider, 1)
        layout.addLayout(prov_row)
        key_row = QHBoxLayout()
        self.edit_ai_key = QLineEdit()
        self.edit_ai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_ai_key.setPlaceholderText("API key (BYOK, optional — env vars also work)")
        key_row.addWidget(QLabel("API key:"))
        key_row.addWidget(self.edit_ai_key, 1)
        layout.addLayout(key_row)

        btn_row = QHBoxLayout()
        self.btn_ai_plan = QPushButton("AI Plan…")
        self.btn_ai_render = QPushButton("Accept && Render AI…")
        btn_row.addWidget(self.btn_ai_plan)
        btn_row.addWidget(self.btn_ai_render)
        layout.addLayout(btn_row)
        self.ai_table = QTableWidget(0, 5)
        self.ai_table.setHorizontalHeaderLabels(
            ["Accept?", "Kind", "Start (s)", "End (s)", "Reason"]
        )
        self.ai_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.ai_table)
        cap_row = QHBoxLayout()
        self.chk_ai_captions = QCheckBox("Burn transcript captions (bilingual)")
        self.chk_ai_captions.setChecked(True)
        cap_row.addWidget(self.chk_ai_captions)
        self.cmb_caption_style = QComboBox()
        from openclip.ai.render_primitives import STYLE_TEMPLATES

        for name in STYLE_TEMPLATES:
            self.cmb_caption_style.addItem(name)
        self.cmb_caption_style.setCurrentText("Classic")
        cap_row.addWidget(QLabel("Caption style:"))
        cap_row.addWidget(self.cmb_caption_style, 1)
        layout.addLayout(cap_row)
        self.chk_auto_reframe = QCheckBox("Auto-Reframe (9:16, subject tracking)")
        self.chk_auto_reframe.setChecked(False)
        self.chk_auto_reframe.setToolTip(
            "Crop a smooth, subject-following 9:16 window before the export scale "
            "(mock tracking data for now)"
        )
        layout.addWidget(self.chk_auto_reframe)
        self.reframe_preview = ReframePreviewWidget()
        layout.addWidget(self.reframe_preview)
        self.chk_auto_reframe.toggled.connect(self._refresh_reframe_preview)

        self.preview = VideoPreviewWidget()
        layout.addWidget(self.preview)

        self.chk_beauty = QCheckBox("Beauty & Retouching")
        self.chk_beauty.setChecked(False)
        if not beauty_available():
            self.chk_beauty.setEnabled(False)
            self.chk_beauty.setToolTip("Requires numpy + opencv (vision extra)")
        else:
            self.chk_beauty.setToolTip(
                "Apply skin smoothing, jaw/neck sculpting, and blemish healing "
                "as a pre-pass before export"
            )
        layout.addWidget(self.chk_beauty)

        skin_row = QHBoxLayout()
        skin_row.addWidget(QLabel("Skin Smoothness:"))
        self.slider_beauty_skin = QSlider(Qt.Orientation.Horizontal)
        self.slider_beauty_skin.setRange(0, 100)
        self.slider_beauty_skin.setValue(15)
        self.slider_beauty_skin.setEnabled(False)
        skin_row.addWidget(self.slider_beauty_skin, 1)
        layout.addLayout(skin_row)

        jaw_row = QHBoxLayout()
        jaw_row.addWidget(QLabel("Jaw Slim:"))
        self.slider_beauty_jaw = QSlider(Qt.Orientation.Horizontal)
        self.slider_beauty_jaw.setRange(0, 100)
        self.slider_beauty_jaw.setValue(10)
        self.slider_beauty_jaw.setEnabled(False)
        jaw_row.addWidget(self.slider_beauty_jaw, 1)
        layout.addLayout(jaw_row)

        neck_row = QHBoxLayout()
        neck_row.addWidget(QLabel("Neck Tighten:"))
        self.slider_beauty_neck = QSlider(Qt.Orientation.Horizontal)
        self.slider_beauty_neck.setRange(0, 100)
        self.slider_beauty_neck.setValue(10)
        self.slider_beauty_neck.setEnabled(False)
        neck_row.addWidget(self.slider_beauty_neck, 1)
        layout.addLayout(neck_row)

        self.chk_beauty.toggled.connect(self._on_beauty_toggled)
        note = QLabel(
            "Checked rows are accepted and rendered; unchecked rows are rejected. "
            "Provider defaults to local_mock (offline); pick a real provider and "
            "paste its API key (BYOK) or set the env var."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.btn_ai_plan.clicked.connect(self.ai_plan)
        self.btn_ai_render.clicked.connect(self.ai_render)
        self.btn_ai_plan.setEnabled(False)
        self.btn_ai_render.setEnabled(False)
        return box

    def _build_moments_panel(self) -> QWidget:
        box = QGroupBox("Top Moments (viral hooks)")
        layout = QVBoxLayout(box)
        btn_row = QHBoxLayout()
        self.btn_find_moments = QPushButton("Find viral moments")
        btn_row.addWidget(self.btn_find_moments)
        btn_row.addWidget(QLabel("Top N:"))
        self.spin_moments_top = QSpinBox()
        self.spin_moments_top.setRange(1, 20)
        self.spin_moments_top.setValue(5)
        btn_row.addWidget(self.spin_moments_top)
        layout.addLayout(btn_row)
        self.moments_table = QTableWidget(0, 6)
        self.moments_table.setHorizontalHeaderLabels(
            ["Accept?", "Start (s)", "End (s)", "Hook type", "Score", "Snippet"]
        )
        self.moments_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.moments_table)
        note = QLabel(
            "Rule-based hook detection over the loaded transcript. Accepted rows "
            "become clip proposals in the AI Director panel and render via "
            "Accept && Render AI…"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.btn_find_moments.clicked.connect(self.find_viral_moments)
        self.btn_find_moments.setEnabled(False)
        return box

    def find_viral_moments(self) -> None:
        """Run hook detection on the loaded transcript and show Top Moments."""
        from openclip.ai import parse_transcript
        from openclip.ai.hook_detection import HookDetector, hooks_to_director_plan

        data = self._original_transcript
        if data is None:
            blocks = [
                b.strip() for b in self.transcript_edit.toPlainText().split("\n\n") if b.strip()
            ]
            data = {"segments": [{"text": b} for b in blocks]}
        segments = parse_transcript(data)
        transcript: list[dict[str, object]] = [
            {"text": s.text, "start": s.start_s, "end": s.end_s} for s in segments
        ]
        hooks = HookDetector(transcript).find_hooks(top_n=self.spin_moments_top.value())
        self._hooks = hooks
        self._populate_moments_table()
        if hooks:
            self._ai_plan = hooks_to_director_plan(
                hooks,
                source_duration_seconds=(
                    self.media.duration_s if self.media is not None else None
                ),
            )
            self._refresh_ai_table()
            self.btn_ai_render.setEnabled(True)
        self.statusBar().showMessage(
            f"Found {len(hooks)} viral moment(s) — review in Top Moments / AI panel.",
            10000,
        )

    def _populate_moments_table(self) -> None:
        self.moments_table.setRowCount(0)
        for row, hook in enumerate(self._hooks):
            self.moments_table.insertRow(row)
            box = QTableWidgetItem()
            box.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            box.setCheckState(Qt.CheckState.Checked)  # proposed accept
            self.moments_table.setItem(row, 0, box)
            for col, text in enumerate(
                (
                    f"{hook.start:.2f}",
                    f"{hook.end:.2f}",
                    hook.hook_type,
                    f"{hook.score:.2f}",
                    hook.transcript_snippet[:80],
                ),
                start=1,
            ):
                cell = QTableWidgetItem(text)
                cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.moments_table.setItem(row, col, cell)

    def _moments_acceptance(self) -> list[bool]:
        result = []
        for row in range(self.moments_table.rowCount()):
            item = self.moments_table.item(row, 0)
            result.append(
                item is not None and item.checkState() == Qt.CheckState.Checked
            )
        return result

    def _build_broll_panel(self) -> QWidget:
        box = QGroupBox("B-roll Proposals")
        layout = QVBoxLayout(box)
        btn_row = QHBoxLayout()
        self.btn_find_broll = QPushButton("Find B-roll")
        btn_row.addWidget(self.btn_find_broll)
        btn_row.addWidget(QLabel("Library:"))
        self.edit_broll_library = QLineEdit()
        self.edit_broll_library.setPlaceholderText("Optional: path to broll_library.json")
        btn_row.addWidget(self.edit_broll_library, 1)
        layout.addLayout(btn_row)
        self.broll_table = QTableWidget(0, 5)
        self.broll_table.setHorizontalHeaderLabels(
            ["Accept?", "Start (s)", "End (s)", "Match Type", "Snippet"]
        )
        self.broll_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.broll_table)
        note = QLabel(
            "Rule-based B-roll detection with local library sourcing. "
            "Accepted rows become B-roll proposals in the AI Director panel "
            "and render via Accept && Render AI…"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.btn_find_broll.clicked.connect(self.find_broll)
        self.btn_find_broll.setEnabled(False)
        return box

    def find_broll(self) -> None:
        """Run B-roll detection on the loaded transcript and show proposals."""
        from openclip.ai import parse_transcript
        from openclip.ai.broll_detection import broll_to_director_plan
        from openclip.ai.transcript_editing import TranscriptSegment

        data = self._original_transcript
        if data is None:
            text_blocks = [
                b.strip() for b in self.transcript_edit.toPlainText().split("\n\n") if b.strip()
            ]
            data = {"segments": [{"text": b} for b in text_blocks]}
        segments = parse_transcript(data)
        timed_blocks: list[TranscriptSegment] = [
            TranscriptSegment(text=s.text, start_s=s.start_s, end_s=s.end_s)
            for s in segments
            if s.start_s is not None and s.end_s is not None
        ]

        detector = BrollDetector(timed_blocks)
        # Exclude hook windows (Face-First rule)
        hook_windows = tuple((h.start, h.end) for h in self._hooks)
        opportunities = detector.find_opportunities(exclude_hooks=hook_windows)

        # Resolve sources
        library_path = self.edit_broll_library.text().strip() or None
        resolver = BrollSourceResolver(library_path)
        sources = [resolver.resolve(opp, library_path=library_path) for opp in opportunities]

        self._broll_opportunities = opportunities
        self._broll_sources = sources
        self._populate_broll_table()

        if opportunities:
            self._ai_plan = broll_to_director_plan(
                opportunities,
                sources,
                source_duration_seconds=self.media.duration_s if self.media is not None else None,
            )
            self._refresh_ai_table()
            self.btn_ai_render.setEnabled(True)

        self.statusBar().showMessage(
            f"Found {len(opportunities)} B-roll opportunity(ies) — "
            "review in B-roll Proposals / AI panel.",
            10000,
        )

    def _populate_broll_table(self) -> None:
        self.broll_table.setRowCount(0)
        for row, (opp, src) in enumerate(
            zip(self._broll_opportunities, self._broll_sources, strict=True)
        ):
            self.broll_table.insertRow(row)
            box = QTableWidgetItem()
            box.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            box.setCheckState(Qt.CheckState.Checked)  # proposed accept
            self.broll_table.setItem(row, 0, box)
            for col, text in enumerate(
                (
                    f"{opp.start:.2f}",
                    f"{opp.end:.2f}",
                    src.match_type,
                    opp.concept_query[:80],
                ),
                start=1,
            ):
                cell = QTableWidgetItem(text)
                cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.broll_table.setItem(row, col, cell)

    def _broll_acceptance(self) -> list[bool]:
        result = []
        for row in range(self.broll_table.rowCount()):
            item = self.broll_table.item(row, 0)
            result.append(
                item is not None and item.checkState() == Qt.CheckState.Checked
            )
        return result

    def _is_hook_plan(self) -> bool:
        from openclip.ai import AIDirectorPlan

        return (
            isinstance(self._ai_plan, AIDirectorPlan)
            and self._ai_plan.provider == "hook_detection"
        )

    def _is_broll_plan(self) -> bool:
        from openclip.ai import AIDirectorPlan

        return (
            isinstance(self._ai_plan, AIDirectorPlan)
            and self._ai_plan.provider == "broll_detection"
        )

    def _refresh_reframe_preview(self, checked: bool) -> None:
        if not checked or self.media is None:
            self.reframe_preview.clear()
            return
        fps = self.media.fps if self.media.fps > 0 else 30.0
        points = tracking_from_default_face_analysis(fps=fps, duration=self.media.duration_s)
        self.reframe_preview.set_tracking(
            points,
            duration=self.media.duration_s,
            src_width=self.media.width,
            src_height=self.media.height,
            target_aspect_ratio=DEFAULT_TARGET_ASPECT,
        )

    def _on_beauty_toggled(self, checked: bool) -> None:
        enabled = checked and beauty_available()
        self.slider_beauty_skin.setEnabled(enabled)
        self.slider_beauty_jaw.setEnabled(enabled)
        self.slider_beauty_neck.setEnabled(enabled)

    def _build_settings_panel(self) -> QWidget:
        box = QGroupBox("Export settings")
        form = QFormLayout(box)

        self.cmb_transcriber = QComboBox()
        for name in available_transcribers():
            self.cmb_transcriber.addItem(name)
        default = "auto" if "whisper" in available_transcribers() else "even"
        self.cmb_transcriber.setCurrentText(default)
        form.addRow("Alignment:", self.cmb_transcriber)

        self.edit_language = QLineEdit()
        self.edit_language.setPlaceholderText("e.g. en (blank = detect)")
        form.addRow("Language:", self.edit_language)

        self.cmb_animation = QComboBox()
        for name in ANIMATIONS:
            self.cmb_animation.addItem(name)
        form.addRow("Caption animation:", self.cmb_animation)

        self.chk_karaoke = QCheckBox("Karaoke word highlight")
        form.addRow("", self.chk_karaoke)

        self.chk_captions = QCheckBox("Burn captions (bilingual)")
        self.chk_captions.setChecked(True)
        form.addRow("", self.chk_captions)

        self.cmb_audio = QComboBox()
        for name in PRESET_NAMES:
            self.cmb_audio.addItem(name)
        self.cmb_audio.setCurrentText("teaching")
        form.addRow("Audio preset:", self.cmb_audio)

        self.edit_crf = QLineEdit("20")
        self.edit_crf.setToolTip("Quality: lower = better/larger (18-26 typical)")
        form.addRow("Video CRF:", self.edit_crf)
        return box

    def _build_player_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(160)
        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio_out)
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(42)
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(lambda pos: self.player.setPosition(pos))
        layout.addWidget(self.btn_play)
        layout.addWidget(self.video_widget, 1)
        layout.addWidget(self.slider)
        self.player.positionChanged.connect(
            lambda pos: self.slider.setValue(pos) if not self.slider.isSliderDown() else None
        )
        return bar

    # ---------- project lifecycle ----------

    def new_project(self) -> None:
        media_path, _ = QFileDialog.getOpenFileName(
            self, "Choose lesson video/audio", "", "Media (*.mp4 *.mov *.mkv *.wav *.m4a)"
        )
        if not media_path:
            return
        try:
            self.media = probe(media_path)
        except Exception as exc:
            QMessageBox.critical(self, "OpenClip", f"Cannot read media:\n{exc}")
            return
        self.project = Project.create(source=media_path, script_text="")
        self._after_project_load()
        self.statusBar().showMessage(
            f"Loaded {Path(media_path).name}: {self.media.width}x{self.media.height} "
            f"{self.media.duration_s:.1f}s"
        )

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "OpenClip project (*.json)")
        if not path:
            return
        try:
            self.project = Project.load(path)
            self.media = probe(self.project.source)
            self.script_edit.setPlainText(
                "\n\n".join(
                    f"{ln.primary} :: {ln.support}" if ln.support else ln.primary
                    for ln in self.project.script.lines
                )
            )
            self._after_project_load()
            self._refresh_cut_table()
        except Exception as exc:
            QMessageBox.critical(self, "OpenClip", f"Cannot open project:\n{exc}")

    def save_project(self) -> None:
        if self.project is None:
            return
        if not self._sync_script_into_project():
            return
        if not hasattr(self, "_project_path") or self._project_path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save project", "lesson.openclip.json", "OpenClip project (*.json)"
            )
            if not path:
                return
            self._project_path = Path(path)
        self.project.save(self._project_path)
        self.statusBar().showMessage(f"Saved {self._project_path}")

    def _after_project_load(self) -> None:
        assert self.project is not None and self.media is not None
        self._project_path = None
        self.project.audio_preset = self.cmb_audio.currentText()
        self.project.captions.animation = self.cmb_animation.currentText()
        self.project.captions.karaoke = self.chk_karaoke.isChecked()
        self.media_label.setText(f"{Path(self.project.source).name} — {self.media.duration_s:.1f}s")
        self.preview.load(str(self.project.source))
        self.player.setSource(QUrl.fromLocalFile(str(self.project.source)))
        self.slider.setRange(0, int(self.media.duration_s * 1000))
        for btn in (
            self.btn_save,
            self.btn_analyze,
            self.btn_export,
            self.btn_ai_plan,
            self.btn_load_transcript,
            self.btn_apply_transcript,
            self.btn_find_moments,
            self.btn_find_broll,
        ):
            btn.setEnabled(True)
        if self.project.script.lines:
            self.transcript_edit.setPlainText(
                "\n\n".join(
                    f"{ln.primary} :: {ln.support}" if ln.support else ln.primary
                    for ln in self.project.script.lines
                )
            )

    # ---------- analysis / review / export ----------

    def _sync_script_into_project(self) -> bool:
        assert self.project is not None
        from openclip.engine.script import parse_script

        self.project.script = parse_script(self.script_edit.toPlainText())
        self.project.align_language = self.edit_language.text().strip() or None
        self.project.captions.animation = self.cmb_animation.currentText()
        self.project.captions.karaoke = self.chk_karaoke.isChecked()
        self.project.audio_preset = self.cmb_audio.currentText()
        return True

    def analyze(self) -> None:
        if self.project is None or self.media is None or self._analysis_runner.busy:
            return
        if not self._sync_script_into_project():
            return
        self.btn_analyze.setEnabled(False)
        self.statusBar().showMessage("Analyzing (align + silence + cut plan)…")
        worker = AnalysisWorker(
            AnalysisRequest(
                project=self.project,
                media=self.media,
                transcriber=self.cmb_transcriber.currentText(),
                language=self.edit_language.text().strip() or None,
            )
        )
        worker.finished.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        worker.note.connect(lambda n: self.statusBar().showMessage(n, 8000))
        self._analysis_runner.start(worker)

    def _on_analysis_done(self, outcome: object) -> None:
        assert self.project is not None
        from openclip.engine.pipeline import PlanOutcome

        assert isinstance(outcome, PlanOutcome)
        self.statusBar().showMessage(
            f"Analyzed: {outcome.transcriber} alignment, "
            f"{len(outcome.cut_plan.cuts)} proposed cuts, "
            f"{outcome.silences_found} silences detected."
        )
        self._refresh_cut_table()
        self.btn_analyze.setEnabled(True)

    def _on_analysis_failed(self, message: str) -> None:
        self.btn_analyze.setEnabled(True)
        QMessageBox.critical(self, "OpenClip", f"Analysis failed:\n{message}")

    def _refresh_cut_table(self) -> None:
        assert self.project is not None
        self.cut_table.setRowCount(0)
        if self.project.cut_plan is None:
            return
        for row, cut in enumerate(self.project.cut_plan.cuts):
            self.cut_table.insertRow(row)
            keep_box = QTableWidgetItem()
            keep_box.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            keep_box.setCheckState(
                Qt.CheckState.Unchecked if cut.accepted else Qt.CheckState.Checked
            )
            self.cut_table.setItem(row, 0, keep_box)
            for col, text in enumerate(
                (
                    cut.kind,
                    f"{cut.start_s:.2f}",
                    f"{cut.end_s:.2f}",
                    cut.reason,
                ),
                start=1,
            ):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.cut_table.setItem(row, col, item)

    def _cut_acceptance_from_table(self) -> None:
        assert self.project is not None and self.project.cut_plan is not None
        for row in range(self.cut_table.rowCount()):
            box = self.cut_table.item(row, 0)
            if box is None or self.project.cut_plan is None:
                continue
            # Unchecked = cut applied (removed). Checked = keep span.
            self.project.cut_plan.cuts[row].accepted = box.checkState() == Qt.CheckState.Unchecked

    def _jump_to_cut(self, row: int, _col: int) -> None:
        assert self.project is not None and self.project.cut_plan is not None
        if row >= len(self.project.cut_plan.cuts):
            return
        cut = self.project.cut_plan.cuts[row]
        self.player.setPosition(int(cut.start_s * 1000))
        self.player.play()

    def export(self) -> None:
        if self.project is None or self.media is None or self._render_runner.busy:
            return
        if self.project.alignment is None:
            QMessageBox.information(self, "OpenClip", "Run Analyze first so the cut plan exists.")
            return
        self._cut_acceptance_from_table()
        out, _ = QFileDialog.getSaveFileName(self, "Export MP4", "lesson-edited.mp4", "MP4 (*.mp4)")
        if not out:
            return
        try:
            crf = int(self.edit_crf.text())
        except ValueError:
            crf = 20
        from openclip.engine.render import RenderOptions

        self.btn_export.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Rendering…")
        worker = RenderWorker(
            RenderRequest(
                project=self.project,
                media=self.media,
                out_path=Path(out),
                options=RenderOptions(
                    crf=crf,
                    burn_captions=self.chk_captions.isChecked(),
                ),
            )
        )
        worker.progress.connect(self._on_render_progress)
        worker.finished.connect(lambda res: self._on_render_done(out, res))
        worker.failed.connect(self._on_render_failed)
        self._render_runner.start(worker)

    def _on_render_progress(self, stage: str, i: int, n: int) -> None:
        self.progress.setRange(0, max(1, n))
        self.progress.setValue(i)
        self.statusBar().showMessage(f"{stage}: {i}/{n}")

    def _on_render_done(self, out_path: str, result: object) -> None:
        self._finish_progress()
        self.btn_export.setEnabled(True)
        assert result is not None
        QMessageBox.information(
            self,
            "Export complete",
            f"Exported to:\n{out_path}\n\n"
            f"Kept {getattr(result, 'kept_duration_s', 0):.1f}s, "
            f"{getattr(result, 'cuts_applied', 0)} cuts applied.",
        )
        self.statusBar().showMessage(f"Exported {out_path}", 10000)

    def _on_render_failed(self, message: str) -> None:
        self._finish_progress()
        self.btn_export.setEnabled(True)
        QMessageBox.critical(self, "OpenClip", f"Export failed:\n{message}")

    def _finish_progress(self) -> None:
        if hasattr(self, "progress"):
            self.statusBar().removeWidget(self.progress)
            self.progress = None  # type: ignore[assignment]

    # ---------- transcript editing ----------

    def load_transcript(self) -> None:
        """Load a timestamped transcript (JSON) or plain text into the editor."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open transcript", "", "Transcript (*.json *.txt);;All files (*)"
        )
        if not path:
            return
        from openclip.ai import parse_transcript

        raw = Path(path).read_text(encoding="utf-8")
        try:
            data: object = json.loads(raw)
        except ValueError:
            data = raw
        self._original_transcript = data
        segments = parse_transcript(data)
        self.transcript_edit.setPlainText("\n\n".join(s.text for s in segments if s.text.strip()))
        self.statusBar().showMessage(f"Loaded transcript {Path(path).name}: {len(segments)} blocks")

    def apply_transcript_edits(self) -> None:
        """Diff the edited transcript against the original → proposed cut decisions.

        The cuts are fed into the AI Director review panel (as a plan whose
        only content is the transcript cuts), so the user sees exactly what
        will be removed before rendering.
        """
        from openclip.ai import AICaption, AICut, AIDirectorPlan, AISpan, transcript_to_edits
        from openclip.director.models import Decision

        original = self._original_transcript
        if original is None:
            # fall back to the script text (no timestamps → warnings, no cuts)
            blocks = [b.strip() for b in self.script_edit.toPlainText().split("\n\n") if b.strip()]
            original = {"segments": [{"text": b} for b in blocks]}
        edited = self.transcript_edit.toPlainText()
        decisions = transcript_to_edits(original, edited, include_captions=True)
        self._transcript_decisions = list(decisions)

        cuts: list[AICut] = []
        captions: list[AICaption] = []
        for d in decisions:
            assert isinstance(d, Decision)
            if d.source_span is None:
                continue
            span = AISpan(d.source_span[0], d.source_span[1])
            if d.kind == "cut":
                cuts.append(AICut(span=span, id=d.id, reason=d.rationale, extra=dict(d.params)))
            elif d.kind == "caption":
                captions.append(
                    AICaption(
                        span=span,
                        id=d.id,
                        primary_text=str(d.params.get("primary") or ""),
                        support_text=(
                            str(d.params["support"]) if d.params.get("support") else None
                        ),
                        extra=dict(d.params),
                    )
                )
        self._ai_plan = AIDirectorPlan(
            version=1,
            provider="transcript_edit",
            prompt="Transcript edits (user-deleted blocks)",
            cuts=cuts,
            captions=captions,
        )
        self._refresh_ai_table()
        self.btn_ai_render.setEnabled(bool(cuts or captions))
        self.statusBar().showMessage(
            f"Transcript edits: {len(cuts)} cut(s), {len(captions)} caption(s) "
            "proposed — review in the AI panel.",
            10000,
        )

    # ---------- AI Director: plan → review → render ----------

    def ai_plan(self) -> None:
        """Run the AI pipeline for the current media + prompt (background thread)."""
        if self.media is None or self._ai_plan_runner.busy:
            return
        prompt = self.edit_ai_prompt.text().strip()
        if not prompt:
            QMessageBox.information(self, "AI Director", "Enter an editing prompt first.")
            return
        self.btn_ai_plan.setEnabled(False)
        self.statusBar().showMessage("AI planning (local_mock, offline)…")
        worker = AIPlanWorker(
            AIPlanRequest(
                media_path=self.media.path,
                prompt=prompt,
                transcript=self.script_edit.toPlainText(),
                provider=self.cmb_ai_provider.currentText(),
                api_key=self.edit_ai_key.text().strip() or None,
            )
        )
        worker.finished.connect(self._on_ai_plan_done)
        worker.failed.connect(self._on_ai_plan_failed)
        self._ai_plan_runner.start(worker)

    def _on_ai_plan_done(self, plan: object) -> None:
        from openclip.ai import AIDirectorPlan

        assert isinstance(plan, AIDirectorPlan)
        self._ai_plan = plan
        self._refresh_ai_table()
        self.btn_ai_plan.setEnabled(True)
        self.btn_ai_render.setEnabled(True)
        self.statusBar().showMessage(
            f"AI plan ready: {len(plan.clips)} clips, {len(plan.cuts)} cuts, "
            f"{len(plan.zooms)} zooms, {len(plan.broll)} b-roll — review below."
        )

    def _on_ai_plan_failed(self, message: str) -> None:
        self.btn_ai_plan.setEnabled(True)
        QMessageBox.critical(self, "AI Director", f"AI planning failed:\n{message}")

    def _refresh_ai_table(self) -> None:
        from openclip.ai import plan_to_review_items

        self.ai_table.setRowCount(0)
        if self._ai_plan is None:
            return
        for row, item in enumerate(plan_to_review_items(self._ai_plan)):  # type: ignore[arg-type]
            self.ai_table.insertRow(row)
            box = QTableWidgetItem()
            box.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            box.setCheckState(Qt.CheckState.Checked)  # PENDING → proposed accept
            self.ai_table.setItem(row, 0, box)
            start = f"{item.start_seconds:.2f}"
            end = f"{item.end_seconds:.2f}"
            for col, text in enumerate((item.kind, start, end, item.reason), start=1):
                cell = QTableWidgetItem(text)
                cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.ai_table.setItem(row, col, cell)

    def _ai_acceptance(self) -> list[bool]:
        result = []
        for row in range(self.ai_table.rowCount()):
            item = self.ai_table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                result.append(True)
            else:
                result.append(False)
        return result

    def ai_render(self) -> None:
        """Render only the accepted AI review items through the director EDL."""
        if self.media is None or self._ai_plan is None or self._ai_render_runner.busy:
            return
        if self.ai_table.rowCount() == 0:
            QMessageBox.information(self, "AI Director", "Run AI Plan… first.")
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Export AI-edited MP4", "lesson-ai.mp4", "MP4 (*.mp4)"
        )
        if not out:
            return
        from openclip.ai import ai_plan_to_director_decisions, plan_to_review_items
        from openclip.director.models import EditDecisionList

        plan = self._ai_plan
        items = plan_to_review_items(plan)  # type: ignore[arg-type]
        if self._is_hook_plan() and len(self._hooks) == self.moments_table.rowCount():
            acceptance = self._moments_acceptance()  # Top Moments checkboxes drive hook clips
        elif (
            self._is_broll_plan()
            and len(self._broll_opportunities) == self.broll_table.rowCount()
        ):
            acceptance = self._broll_acceptance()  # B-roll Proposals checkboxes drive broll
        else:
            acceptance = self._ai_acceptance()
        accepted_ids = {item.id for item, ok in zip(items, acceptance, strict=True) if ok}
        decisions = ai_plan_to_director_decisions(plan)  # type: ignore[arg-type]
        for decision in decisions:
            decision.status = "accepted" if decision.id in accepted_ids else "rejected"
        project = self.project or Project.create(source=self.media.path, script_text="")
        edl = EditDecisionList(project_source=str(self.media.path), decisions=decisions)
        self.btn_ai_render.setEnabled(False)
        self.statusBar().showMessage("Rendering AI plan…")
        try:
            crf = int(self.edit_crf.text())
        except ValueError:
            crf = 20
        beauty = None
        if self.chk_beauty.isChecked():
            beauty = BeautyConfig(
                skin_smooth=self.slider_beauty_skin.value() / 100.0,
                jaw_slim=self.slider_beauty_jaw.value() / 100.0,
                neck_tighten=self.slider_beauty_neck.value() / 100.0,
            )
        worker = AIRenderWorker(
            AIRenderRequest(
                project=project,
                media=self.media,
                edl=edl,
                out_path=Path(out),
                burn_captions=self.chk_captions.isChecked(),
                crf=crf,
                transcript_captions=self.chk_ai_captions.isChecked(),
                caption_style=self.cmb_caption_style.currentText() or None,
                smart_reframe=self.chk_auto_reframe.isChecked(),
                beauty=beauty,
            )
        )
        worker.progress.connect(self._on_render_progress)
        worker.finished.connect(lambda res: self._on_ai_render_done(out, res))
        worker.failed.connect(self._on_ai_render_failed)
        self._ai_render_runner.start(worker)

    def _on_ai_render_done(self, out_path: str, result: object) -> None:
        self._finish_progress()
        self.btn_ai_render.setEnabled(True)
        QMessageBox.information(
            self,
            "AI export complete",
            f"Exported to:\n{out_path}\n\n"
            f"Kept {getattr(result, 'kept_duration_s', 0):.1f}s, "
            f"{getattr(result, 'cuts_applied', 0)} cuts applied.",
        )
        self.statusBar().showMessage(f"Exported {out_path}", 10000)

    def _on_ai_render_failed(self, message: str) -> None:
        self._finish_progress()
        self.btn_ai_render.setEnabled(True)
        QMessageBox.critical(self, "AI Director", f"AI render failed:\n{message}")

    # ---------- playback ----------

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

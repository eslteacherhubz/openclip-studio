"""Main window: project open/create, script editing, cut review, export."""

from __future__ import annotations

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
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from openclip.engine.captions import ANIMATIONS
from openclip.engine.enhance import PRESET_NAMES
from openclip.engine.ffmpeg import MediaInfo, probe
from openclip.engine.pipeline import available_transcribers
from openclip.engine.project import Project
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

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addLayout(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        splitter.addWidget(self._build_script_panel())
        splitter.addWidget(self._build_review_panel())
        splitter.addWidget(self._build_ai_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.setSizes([340, 360, 300, 260])

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
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", "OpenClip project (*.json)"
        )
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
        self.media_label.setText(
            f"{Path(self.project.source).name} — {self.media.duration_s:.1f}s"
        )
        self.player.setSource(QUrl.fromLocalFile(str(self.project.source)))
        self.slider.setRange(0, int(self.media.duration_s * 1000))
        for btn in (self.btn_save, self.btn_analyze, self.btn_export, self.btn_ai_plan):
            btn.setEnabled(True)

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
            self.project.cut_plan.cuts[row].accepted = (
                box.checkState() == Qt.CheckState.Unchecked
            )

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
            QMessageBox.information(
                self, "OpenClip", "Run Analyze first so the cut plan exists."
            )
            return
        self._cut_acceptance_from_table()
        out, _ = QFileDialog.getSaveFileName(
            self, "Export MP4", "lesson-edited.mp4", "MP4 (*.mp4)"
        )
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
        return [
            (self.ai_table.item(row, 0) is not None)
            and self.ai_table.item(row, 0).checkState() == Qt.CheckState.Checked
            for row in range(self.ai_table.rowCount())
        ]

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
        accepted_ids = {
            item.id for item, ok in zip(items, self._ai_acceptance(), strict=True) if ok
        }
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
        worker = AIRenderWorker(
            AIRenderRequest(
                project=project,
                media=self.media,
                edl=edl,
                out_path=Path(out),
                burn_captions=self.chk_captions.isChecked(),
                crf=crf,
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

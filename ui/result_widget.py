from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.facecolor'] = '#1a1a2e'
matplotlib.rcParams['axes.facecolor'] = '#16213e'
matplotlib.rcParams['axes.edgecolor'] = '#2c3e6b'
matplotlib.rcParams['axes.labelcolor'] = '#a0b4d0'
matplotlib.rcParams['text.color'] = '#a0b4d0'
matplotlib.rcParams['xtick.color'] = '#5a7a9a'
matplotlib.rcParams['ytick.color'] = '#5a7a9a'
matplotlib.rcParams['legend.facecolor'] = '#1a2744'
matplotlib.rcParams['legend.edgecolor'] = '#2c3e6b'


DARK_CARD = 'background: #1e2a3e; border-radius: 8px; padding: 12px;'
ACCENT = '#0fbc9c'
WARN = '#f39c12'
TEXT_LIGHT = '#c8d6e5'
TEXT_MUTED = '#7f8fa6'

METRIC_TEMPLATE = (
    'background: #16213e; border-radius: 8px; padding: 10px; '
    'border: 1px solid #1e2a4a;'
)
VALUE_TEMPLATE = (
    'font-size: 22px; font-weight: bold; color: {color};'
)
LABEL_TEMPLATE = (
    'font-size: 11px; color: #7f8fa6; margin-top: 2px;'
)


def metric_card(value, label, color=ACCENT):
    card = QFrame()
    card.setStyleSheet(METRIC_TEMPLATE)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(2)
    v = QLabel(str(value))
    v.setStyleSheet(VALUE_TEMPLATE.format(color=color))
    v.setAlignment(Qt.AlignCenter)
    layout.addWidget(v)
    l = QLabel(label)
    l.setStyleSheet(LABEL_TEMPLATE)
    l.setAlignment(Qt.AlignCenter)
    layout.addWidget(l)
    return card


class ResultWidget(QWidget):
    back_requested = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background: #1a1a2e;')
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 12, 16, 12)

        self.header = QLabel('心率分析结果')
        self.header.setStyleSheet(
            'font-size: 20px; font-weight: bold; color: #c8d6e5; padding: 4px 0;')
        self.header.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.header)

        self.metrics_row = QHBoxLayout()
        self.metrics_row.setSpacing(10)
        layout.addLayout(self.metrics_row)

        self.fig = Figure(figsize=(10, 4.5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet('background: transparent;')
        layout.addWidget(self.canvas, stretch=1)

        self.advice_label = QLabel()
        self.advice_label.setStyleSheet(
            'background: #0d2137; color: #7fb3d8; padding: 14px 16px; '
            'border-radius: 8px; font-size: 13px; '
            'border-left: 3px solid #0fbc9c;')
        self.advice_label.setWordWrap(True)
        self.advice_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.advice_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.back_btn = QPushButton('← 返回首页')
        self.back_btn.setStyleSheet(
            'QPushButton { background: #1e2a3e; color: #7f8fa6; border: 1px solid #2c3e6b; '
            'border-radius: 6px; padding: 8px 20px; font-size: 13px; }'
            'QPushButton:hover { background: #2c3e6b; color: #c8d6e5; border-color: #4a6a9a; }')
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.clicked.connect(self.back_requested.emit)
        btn_row.addWidget(self.back_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def show_analyzing(self):
        self.header.setText('正在分析心率数据...')
        self._clear_metrics()
        self.fig.clear()
        self.canvas.draw()
        self.advice_label.setVisible(True)
        self.advice_label.setText('⏳ 正在处理 1600 帧数据并生成 AI 建议...')

    def _clear_metrics(self):
        while self.metrics_row.count():
            item = self.metrics_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def set_metrics(self, results):
        self._clear_metrics()
        bpm = results['bpm_peaks']
        bpm_color = ACCENT if 60 <= bpm <= 90 else WARN
        lfhf = results['lf_hf_ratio']
        lfhf_color = ACCENT if lfhf < 1 else (WARN if lfhf < 2 else '#e74c3c')
        cards = [
            metric_card(bpm, f'BPM  |  脉搏波峰 {results["peak_count"]}个', bpm_color),
            metric_card(f'{results["bpm_fft"]}', 'BPM (FFT)', TEXT_MUTED),
            metric_card(f'{lfhf:.2f}', 'LF/HF 比值', lfhf_color),
            metric_card(f'{results["lf_power"]:.0f}', 'LF 功率', TEXT_MUTED),
            metric_card(f'{results["hf_power"]:.0f}', 'HF 功率', ACCENT),
        ]
        for c in cards:
            self.metrics_row.addWidget(c)

    def show_advice_loading(self):
        self.advice_label.setVisible(True)
        self.advice_label.setText('⏳ AI 正在分析数据并生成个性化建议...')

    def show_advice(self, text):
        self.advice_label.setText('💡  ' + text)
        self.advice_label.setVisible(True)

    def plot_results(self, bvp_raw, bvp_filt, peaks, freqs, fft_vals, results):
        self.header.setText('心率分析结果')
        self.set_metrics(results)

        self.fig.clear()
        axes = self.fig.subplots(3, 1)
        t = np.arange(len(bvp_raw)) / 30

        line_c = '#0fbc9c'
        axes[0].plot(t, bvp_raw, color=line_c, linewidth=0.7, alpha=0.7)
        axes[0].set_title('BVP 脉搏波', fontsize=10, color='#a0b4d0')
        axes[0].set_ylabel('幅值', color='#7f8fa6')
        axes[0].tick_params(colors='#5a7a9a', labelsize=8)
        axes[0].grid(alpha=0.1, color='#2c3e6b')

        axes[1].plot(t, bvp_filt, color='#f39c12', linewidth=0.7, alpha=0.7)
        axes[1].plot(t[peaks], bvp_filt[peaks], 'x', color='#e74c3c',
                     markersize=4, label=f'波峰 ({len(peaks)}个)')
        axes[1].set_title('滤波后 BVP (0.04-4Hz)', fontsize=10, color='#a0b4d0')
        axes[1].set_ylabel('幅值', color='#7f8fa6')
        axes[1].tick_params(colors='#5a7a9a', labelsize=8)
        axes[1].legend(fontsize=8, labelcolor='#a0b4d0')
        axes[1].grid(alpha=0.1, color='#2c3e6b')

        axes[2].plot(freqs * 60, fft_vals, color='#9b59b6', linewidth=0.9)
        axes[2].axvspan(0.04 * 60, 0.15 * 60, alpha=0.08, color='#e67e22')
        axes[2].axvspan(0.15 * 60, 0.4 * 60, alpha=0.08, color='#2ecc71')
        axes[2].set_title('FFT 频谱', fontsize=10, color='#a0b4d0')
        axes[2].set_xlabel('频率 (BPM)', color='#7f8fa6')
        axes[2].set_ylabel('幅值', color='#7f8fa6')
        axes[2].set_xlim(0, 180)
        axes[2].tick_params(colors='#5a7a9a', labelsize=8)
        axes[2].grid(alpha=0.1, color='#2c3e6b')

        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()

        self.show_advice_loading()

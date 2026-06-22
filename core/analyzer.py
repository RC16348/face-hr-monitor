import numpy as np
from scipy import signal

def bandpass_filter(data, lowcut=0.04, highcut=4.0, fs=30, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, data)

def bpm_from_peaks(bvp, fs=30, min_dist=15):
    peaks, _ = signal.find_peaks(bvp, distance=min_dist)
    if len(peaks) < 2:
        return 0.0, peaks
    intervals = np.diff(peaks) / fs
    return 60.0 / np.mean(intervals), peaks

def bpm_from_fft(bvp, fs=30, low_bpm=30, high_bpm=220):
    n = len(bvp)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_vals = np.abs(np.fft.rfft(bvp))
    valid = (freqs >= low_bpm / 60) & (freqs <= high_bpm / 60)
    if not np.any(valid):
        return 0.0, freqs, fft_vals
    peak_idx = np.argmax(fft_vals * valid)
    return freqs[peak_idx] * 60, freqs, fft_vals

def compute_hrv(bvp, fs=30):
    bvp_detrend = bvp - np.mean(bvp)
    bvp_filt = bandpass_filter(bvp_detrend, fs=fs)

    bpm_p, peaks = bpm_from_peaks(bvp_filt, fs=fs)
    bpm_f, freqs, fft_vals = bpm_from_fft(bvp_filt, fs=fs)

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)

    lf_power = np.trapz(fft_vals[lf_mask] ** 2, freqs[lf_mask]) if np.any(lf_mask) else 0
    hf_power = np.trapz(fft_vals[hf_mask] ** 2, freqs[hf_mask]) if np.any(hf_mask) else 0
    lf_hf = lf_power / hf_power if hf_power > 0 else float('inf')

    return {
        'bpm_peaks': round(bpm_p, 1),
        'bpm_fft': round(bpm_f, 1),
        'peak_count': len(peaks),
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': round(lf_hf, 2),
        'bvp_filtered': bvp_filt,
        'freqs': freqs,
        'fft_vals': fft_vals,
    }

def interpret(results):
    lfhf = results['lf_hf_ratio']
    bpm = results['bpm_peaks'] if results['bpm_peaks'] > 0 else results['bpm_fft']
    lines = []

    if lfhf == float('inf'):
        lines.append(('状态', 'HF 功率极低，自主神经调节可能严重耗竭'))
    elif lfhf < 1:
        lines.append(('状态', '放松状态 — 副交感主导，压力低'))
    elif lfhf < 2:
        lines.append(('状态', '轻度压力 / 轻微疲惫'))
    else:
        lines.append(('状态', '高度紧张 / 焦虑 / 重度疲劳 — 交感神经过度兴奋'))

    if bpm < 60:
        lines.append(('心率', f'{bpm} BPM — 心率偏缓'))
    elif bpm < 80:
        lines.append(('心率', f'{bpm} BPM — 静息正常范围'))
    elif bpm < 100:
        lines.append(('心率', f'{bpm} BPM — 静息偏高，可能存在压力或疲劳'))
    else:
        lines.append(('心率', f'{bpm} BPM — 心率过高'))

    if results['hf_power'] > 0.1:
        lines.append(('恢复', 'HF 功率高，副交感活跃，恢复良好'))
    elif results['hf_power'] > 0.03:
        lines.append(('恢复', 'HF 功率中等，恢复能力一般'))
    else:
        lines.append(('恢复', 'HF 功率偏低，建议休息'))

    return lines

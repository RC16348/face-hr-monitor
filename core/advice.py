from agnes_client import chat

PROMPT = """你是一位专业的健康顾问。请根据以下心率变异性(HRV)分析结果，用中文给出简短、中肯的健康建议。

分析数据（采集时长约53秒，以下为参考范围）：
- BPM（峰值检测）: {bpm_peaks}　正常静息范围 60-100
- BPM（FFT主频）: {bpm_fft}　正常静息范围 60-100
- LF/HF比值: {lf_hf}　正常范围 1-2（<1偏放松，>2偏压力/疲劳）
- LF功率: {lf_power:.1f}　反映交感神经活跃度
- HF功率: {hf_power:.1f}　反映副交感神经活跃度
- 脉搏波峰数: {peak_count}（53秒内检出，正常静息约40-80个）

要求：
1. 用2-3句话解读当前状态（放松/压力/疲劳等）
2. 结合数据给出1-2条具体可操作的建议
3. 语气温和专业，不要恐吓
4. 总字数控制在150字以内"""


def get_advice(results, timeout=30):
    prompt = PROMPT.format(
        bpm_peaks=results['bpm_peaks'],
        bpm_fft=results['bpm_fft'],
        lf_hf=results['lf_hf_ratio'],
        lf_power=results['lf_power'],
        hf_power=results['hf_power'],
        peak_count=results['peak_count'],
    )
    resp = chat([{'role': 'user', 'content': prompt}])
    return resp['choices'][0]['message']['content']

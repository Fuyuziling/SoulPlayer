# -*- coding: utf-8 -*-
"""台词触发方案对比验证脚本（阶段 2）

对比两种特征在"异词区分度"上的表现：
  A. MFCC + DTW（现有 voice_match.py 方案）
  B. wav2vec2 / 自监督内容嵌入 + 余弦相似度（中期候选方案）

对 voice_templates/*.wav 现有模板两两比对：
  - 同词（同句台词的两次录音）距离应小、异词距离应大；
  - 区分度 = 异词距离中位数 - 同词距离中位数（越大越好）。

嵌入部分依赖 torch + transformers + soundfile + wav2vec2 模型：
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers soundfile
模型 wav2vec2-base（约 300MB，首次运行自动下载）。
未安装依赖时，本脚本自动跳过嵌入部分，仅输出 MFCC-DTW 基线。
"""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from voice_match import mfcc, dtw_distance, trim_silence, rms_normalize, SAMPLE_RATE

TEMPLATES_DIR = Path(__file__).parent / "voice_templates"


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, f"{path.name} 采样率不是 {SAMPLE_RATE}"
        n = wf.getnframes()
        pcm = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return rms_normalize(trim_silence(pcm))


def mfcc_dtw_matrix(feats: dict) -> dict:
    """两两 MFCC-DTW 距离矩阵（对角=同词，非对角=异词）。"""
    names = list(feats.keys())
    dist = {}
    for i in range(len(names)):
        for j in range(i, len(names)):
            a, b = names[i], names[j]
            d = dtw_distance(feats[a], feats[b])
            dist[(a, b)] = d
    return dist


def embed_matrix(samples: dict) -> dict:
    """两两 wav2vec2 嵌入余弦距离（1 - cos，与 DTW 同尺度 0~2）。"""
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    device = "cpu"
    model_name = "facebook/wav2vec2-base"
    print(f"[嵌入] 加载 {model_name}（首次运行会下载约 300MB 模型）...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()

    emb = {}
    with torch.no_grad():
        for name, x in samples.items():
            inputs = processor(x, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt").input_values
            out = model(inputs).last_hidden_state  # (1, T, D)
            emb[name] = out.mean(dim=1).squeeze(0).numpy()  # 均值池化

    names = list(emb.keys())
    dist = {}
    for i in range(len(names)):
        for j in range(i, len(names)):
            a, b = names[i], names[j]
            va, vb = emb[a], emb[b]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1e-10 or nb < 1e-10:
                d = 1.0
            else:
                d = 1.0 - float(np.dot(va, vb)) / (na * nb)
            dist[(a, b)] = d
    return dist


def summarize(title: str, dist: dict, names: list):
    print(f"\n===== {title} =====")
    same, diff = [], []
    for i in range(len(names)):
        for j in range(i, len(names)):
            d = dist[(names[i], names[j])]
            tag = "同词" if i == j else "异词"
            print(f"  {tag}  {names[i]} vs {names[j]} = {d:.3f}")
            (same if i == j else diff).append(d)
    if same:
        print(f"  同词距离: min={min(same):.3f} max={max(same):.3f} 中位={np.median(same):.3f}")
    print(f"  异词距离: min={min(diff):.3f} max={max(diff):.3f} 中位={np.median(diff):.3f}")
    if same:
        print(f"  ▶ 区分度（异词中位 - 同词中位）= {np.median(diff) - np.median(same):.3f}（越大越好）")
    return np.median(diff) - (np.median(same) if same else 0.0)


def main():
    wavs = sorted(TEMPLATES_DIR.glob("*.wav"))
    if len(wavs) < 2:
        print(f"voice_templates 下至少需要 2 个模板录音，当前 {len(wavs)} 个")
        return

    print(f"[加载] 找到 {len(wavs)} 个模板录音:")
    for w in wavs:
        print(f"       {w.name}")

    samples = {w.stem: load_wav(w) for w in wavs}
    feats = {name: mfcc(x) for name, x in samples.items()}

    d_mfcc = mfcc_dtw_matrix(feats)
    gap_mfcc = summarize("A. MFCC + DTW（现状）", d_mfcc, list(feats.keys()))

    # 嵌入部分：依赖检测
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        print("\n[嵌入] 未安装 torch/transformers/soundfile，跳过 wav2vec2 对比。")
        print("       如需嵌入对比，请安装：")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("       pip install transformers soundfile")
        return

    try:
        d_emb = embed_matrix(samples)
        gap_emb = summarize("B. wav2vec2 嵌入 + 余弦", d_emb, list(samples.keys()))
        print(f"\n===== 结论 =====")
        print(f"  区分度（异词中位 − 同词中位）: MFCC-DTW={gap_mfcc:.3f}  vs  嵌入={gap_emb:.3f}")
        print(f"  嵌入方案 {'优于' if gap_emb > gap_mfcc else '未优于'} 现状")
    except Exception as e:
        print(f"\n[嵌入] 运行失败: {e}")


if __name__ == "__main__":
    main()

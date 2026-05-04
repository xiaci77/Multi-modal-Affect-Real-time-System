import os
import sys
import subprocess
import argparse
import pandas as pd
import json
import time
from tqdm import tqdm

# ============ 默认路径（可通过命令行参数覆盖）===========
_DEFAULT_BASE = os.environ.get("MM_PROCESS_DIR", os.path.join(os.path.dirname(__file__), "Multimodal", "mm-process"))
DEFAULT_MP4_ROOT = os.environ.get("MP4_ROOT", os.path.join(os.path.dirname(__file__), "Multimodal", "mp4"))
DEFAULT_MM_CSV   = os.path.join(_DEFAULT_BASE, "mm.csv")
DEFAULT_TRANS_OUT = os.path.join(_DEFAULT_BASE, "transcription.csv")
DEFAULT_AUDIO_OUT = os.path.join(_DEFAULT_BASE, "audio.csv")
DEFAULT_CHECKPOINT = os.path.join(_DEFAULT_BASE, "asr_checkpoint.json")
# ======================================================


def extract_audio(mp4_path, wav_path):
    """用 ffmpeg 从 mp4 提取音频为 16kHz 单声道 wav"""
    cmd = [
        'ffmpeg', '-y', '-i', mp4_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='ASR 语音转录 + 音频提取')
    parser.add_argument('--mp4_root', type=str, default=DEFAULT_MP4_ROOT, help='mp4 视频根目录')
    parser.add_argument('--mm_csv', type=str, default=DEFAULT_MM_CSV, help='mm.csv 路径')
    parser.add_argument('--trans_out', type=str, default=DEFAULT_TRANS_OUT, help='transcription.csv 输出路径')
    parser.add_argument('--audio_out', type=str, default=DEFAULT_AUDIO_OUT, help='audio.csv 输出路径')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='断点续传文件路径')
    args = parser.parse_args()

    mp4_root = args.mp4_root
    mm_csv = args.mm_csv
    trans_out = args.trans_out
    audio_out = args.audio_out
    checkpoint_path = args.checkpoint

    print(f"mp4_root:    {mp4_root}")
    print(f"mm_csv:      {mm_csv}")
    print(f"trans_out:   {trans_out}")
    print(f"audio_out:   {audio_out}")
    print(f"checkpoint:  {checkpoint_path}")

    # 断点续传工具函数
    def load_checkpoint():
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_checkpoint(done):
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(done, f, ensure_ascii=False)

    # 检查 ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("错误: 找不到 ffmpeg！请先安装 ffmpeg 并加入 PATH")
        print("  下载地址: https://ffmpeg.org/download.html")
        print("  或者: conda install ffmpeg")
        sys.exit(1)

    # 读取 mm.csv
    df = pd.read_csv(mm_csv)
    print(f"mm.csv 共 {len(df)} 条记录")

    # 构建任务列表
    tasks = []
    for _, row in df.iterrows():
        fname = str(row['file_name'])            # e.g. G00001/G00002/xxx/xxx.mp4
        base = fname.rsplit('.', 1)[0]           # e.g. G00001/G00002/xxx/xxx
        mp4_path = os.path.join(mp4_root, fname.replace('/', os.sep))
        wav_name = fname.replace('.mp4', '.wav') # e.g. G00001/G00002/xxx/xxx.wav
        wav_path = os.path.join(mp4_root, wav_name.replace('/', os.sep))

        if os.path.exists(mp4_path):
            tasks.append((base, mp4_path, wav_path, wav_name))
        else:
            print(f"  [警告] mp4 不存在: {mp4_path}")

    print(f"找到 {len(tasks)} 个 mp4 文件")

    # 统计已有 wav
    existing_wav = sum(1 for _, _, wp, _ in tasks if os.path.exists(wp))
    print(f"已有 wav 文件: {existing_wav}")

    # 加载断点
    done = load_checkpoint()
    print(f"已有转录结果: {len(done)} 条")

    # 需要处理的：没有转录结果的（不管 wav 是否已存在）
    remaining = [(b, mp, wp, wn) for b, mp, wp, wn in tasks if b not in done]
    print(f"待处理: {len(remaining)} 条\n")

    if len(remaining) == 0:
        print("所有文件已处理完成！")
        save_results(done, tasks)
        return

    # 加载 FunASR 模型
    print("正在加载 FunASR Paraformer 模型（首次运行会自动下载，约 1-2GB）...")
    try:
        from funasr import AutoModel
    except ImportError:
        print("错误: 请先安装 FunASR:")
        print("  pip install funasr modelscope")
        sys.exit(1)

    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc",
        punc_model_revision="v2.0.4",
        device="cuda" if __import__('torch').cuda.is_available() else "cpu",
    )
    print("模型加载完成！\n")

    # 逐个处理
    start_time = time.time()
    errors = 0

    try:
        for i, (base_name, mp4_path, wav_path, wav_name) in enumerate(tqdm(remaining, desc="处理中")):
            try:
                # 提取音频（如果 wav 不存在）
                if not os.path.exists(wav_path):
                    if not extract_audio(mp4_path, wav_path):
                        tqdm.write(f"  [跳过] 音频提取失败: {base_name}")
                        errors += 1
                        done[base_name] = ''
                        continue

                # ASR 转录
                res = model.generate(input=wav_path)
                text = res[0]['text'] if res and len(res) > 0 else ''
                done[base_name] = text

            except Exception as e:
                tqdm.write(f"  [错误] {base_name}: {e}")
                errors += 1
                done[base_name] = ''

            # 每 100 条保存断点
            if (i + 1) % 100 == 0:
                save_checkpoint(done)
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                remaining_est = (len(remaining) - i - 1) / speed if speed > 0 else 0
                tqdm.write(f"  进度: {i+1}/{len(remaining)}, "
                           f"速度: {speed:.1f} 条/秒, "
                           f"预计剩余: {remaining_est/60:.1f} 分钟")

    except KeyboardInterrupt:
        print("\n\n用户中断！已保存断点，重新运行会继续。")

    finally:
        save_checkpoint(done)
        save_results(done, tasks, trans_out, audio_out)

    elapsed = time.time() - start_time
    print(f"\n处理完成！")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"  成功: {len(done) - errors}, 失败: {errors}")


def save_results(done, tasks, trans_out, audio_out):
    """保存 transcription.csv 和 audio.csv"""
    # transcription.csv
    trans_records = []
    for base, _, _, wav_name in tasks:
        trans_records.append({'name': base, 'chinese': done.get(base, '')})
    trans_df = pd.DataFrame(trans_records)
    trans_df.to_csv(trans_out, index=False, encoding='utf-8-sig')
    non_empty = (trans_df['chinese'].str.strip() != '').sum()
    print(f"\ntranscription.csv 已保存: {trans_out}")
    print(f"  总条数: {len(trans_df)}, 有文本: {non_empty}, 空文本: {len(trans_df) - non_empty}")

    # audio.csv
    audio_records = []
    for _, _, _, wav_name in tasks:
        audio_records.append({'file_name': wav_name})
    pd.DataFrame(audio_records).to_csv(audio_out, index=False)
    print(f"audio.csv 已保存: {audio_out} ({len(audio_records)} 条)")

    # 打印几条转录示例
    print("\n转录示例:")
    shown = 0
    for _, row in trans_df[trans_df['chinese'].str.strip() != ''].head(5).iterrows():
        text = row['chinese']
        print(f"  {row['name']}: {text[:80]}{'...' if len(text) > 80 else ''}")
        shown += 1
    if shown == 0:
        print("  （暂无转录结果）")


if __name__ == '__main__':
    main()
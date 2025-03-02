import mne
import numpy as np
import os


def load_edf(edf_path, standard_channels=None):
    """读取EDF文件并标准化通道"""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # 统一通道：选择标准通道（例如CHB-MIT的23通道）
    if standard_channels:
        available_channels = [ch for ch in standard_channels if ch in raw.info['ch_names']]
        raw.pick(available_channels)

    # 降采样（可选）：若原始采样率过高（如512Hz），可降低到256Hz
    if raw.info['sfreq'] > 256:
        raw.resample(256)

    data = raw.get_data()  # 形状：(n_channels, n_samples)
    times = raw.times  # 时间轴（单位：秒）
    return data, times, raw.info


# # 示例：读取单个EDF文件
# edf_path = "data/CHB-MIT/chb01/chb01_03.edf"
# standard_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8']  # 完整23通道列表
# data, times, info = load_edf(edf_path, standard_channels)
# print("数据形状:", data.shape)  # 输出：(23, 256*60*60) → 假设1小时数据

def extract_segments(edf_path, summary_path, preictal_window=1800, interictal_window=1800):
    """从EDF文件中提取发作前期和发作间期段"""
    # 读取EDF数据
    data, times, info = load_edf(edf_path)
    fs = int(info['sfreq'])  # 采样率

    # 解析summary文件获取发作时间
    seizure_times = parse_summary(summary_path, os.path.basename(edf_path))

    segments = []
    labels = []

    # 提取发作前期（Preictal）
    for start, end in seizure_times:
        preictal_start = max(0, start - preictal_window)
        # print("preictal_start * fs", preictal_start * fs)
        preictal_data = data[:, int(preictal_start * fs): int(start * fs)]
        if preictal_data.shape[1] >= fs * 600:  # 至少10分钟数据
            segments.append(preictal_data)
            labels.append(1)

    # 提取发作间期（Interictal）：从无发作区域随机截取
    if not seizure_times:
        # 整个文件作为Interictal
        segments.append(data)
        labels.append(0)
    else:
        # 随机选择一个远离发作的时间段
        safe_start = np.random.randint(0, data.shape[1] - interictal_window * fs)
        interictal_data = data[:, safe_start: safe_start + interictal_window * fs]
        segments.append(interictal_data)
        labels.append(0)

    return segments, labels


def parse_summary(summary_path, target_edf):
    """解析summary文件，返回指定EDF文件的发作时间列表"""
    seizure_times = []
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("File Name:") and target_edf in line:
                # 查找后续的发作时间
                j = i + 1
                while j < len(lines):
                    if "Number of Seizures in File:" in lines[j]:
                        num_seizures = int(lines[j].split(":")[1].strip())
                        for k in range(num_seizures):
                            start_line = lines[j + 1 + 2 * k]
                            end_line = lines[j + 2 + 2 * k]
                            start = float(start_line.split(":")[1].replace("seconds", "").strip())
                            end = float(end_line.split(":")[1].replace("seconds", "").strip())
                            seizure_times.append((start, end))
                        break
                    j += 1
    return seizure_times


import h5py
from tqdm import tqdm


def save_to_hdf5(all_segments, all_labels, output_path):
    """将数据保存为HDF5文件，支持大规模存储"""
    with h5py.File(output_path, 'w') as hf:
        # 创建可扩展的数据集
        max_shape = (None, all_segments[0].shape[0], all_segments[0].shape[1])
        hf.create_dataset('data', shape=(0, *max_shape[1:]), maxshape=max_shape, chunks=True)
        hf.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=np.int32)

        for seg, lab in tqdm(zip(all_segments, all_labels), desc="保存数据"):
            # 扩展数据集
            hf['data'].resize((hf['data'].shape[0] + 1), axis=0)
            hf['data'][-1] = seg
            hf['labels'].resize((hf['labels'].shape[0] + 1), axis=0)
            hf['labels'][-1] = lab


# # 示例：批量处理并保存
# all_segments = []
# all_labels = []
# for edf_file in edf_files:
#     segments, labels = extract_segments(edf_file, summary_file)
#     all_segments.extend(segments)
#     all_labels.extend(labels)
#
# save_to_hdf5(all_segments, all_labels, "processed_data.h5")

def process_patient(subject_dir):
    """处理单个患者的所有EDF文件，并返回患者ID、数据片段和标签"""
    patient_id = os.path.basename(subject_dir)
    # print("patient_id", patient_id)
    summary_path = os.path.join(subject_dir, f"chb{patient_id[3:]}-summary.txt")
    # print(summary_path)
    edf_files = [f for f in os.listdir(subject_dir) if f.endswith(".edf")]
    # print(edf_files)

    patient_segments = []
    patient_labels = []
    for edf_file in edf_files:
        edf_path = os.path.join(subject_dir, edf_file)
        # print("edf_path", edf_path)
        segments, labels = extract_segments(edf_path, summary_path)
        patient_segments.extend(segments)
        patient_labels.extend(labels)

    return patient_id, patient_segments, patient_labels

def pad_or_truncate(seg, desired_length):
    """
    如果 EEG 片段 seg 的长度大于 desired_length，则截断；
    如果小于 desired_length，则在后面用0填充；如果等于，直接返回。
    """
    n_channels, current_length = seg.shape
    if current_length > desired_length:
        return seg[:, :desired_length]
    elif current_length < desired_length:
        pad_width = desired_length - current_length
        padding = np.zeros((n_channels, pad_width))
        return np.concatenate([seg, padding], axis=1)
    else:
        return seg



standard_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8']  # 完整23通道列表


# 并行处理所有患者
# subjects = ["data/CHB-MIT/chb01", "data/CHB-MIT/chb02", "data/CHB-MIT/chb03", "data/CHB-MIT/chb04", "data/CHB-MIT/chb05", "data/CHB-MIT/chb06",
#             "data/CHB-MIT/chb07", "data/CHB-MIT/chb08", "data/CHB-MIT/chb09", "data/CHB-MIT/chb10", "data/CHB-MIT/chb11", "data/CHB-MIT/chb12",
#             "data/CHB-MIT/chb13", "data/CHB-MIT/chb14", "data/CHB-MIT/chb15", "data/CHB-MIT/chb16", "data/CHB-MIT/chb17", "data/CHB-MIT/chb18",
#             "data/CHB-MIT/chb19", "data/CHB-MIT/chb20", "data/CHB-MIT/chb21", "data/CHB-MIT/chb22", "data/CHB-MIT/chb23", "data/CHB-MIT/

subjects = ["data/CHB-MIT/chb07"]
desired_length = 30 * 60 * 256

results = [process_patient(subject) for subject in subjects]
print("results", results)

# 针对每个患者单独保存数据
for patient_id, segs, labs in results:
    fixed_segs = [pad_or_truncate(seg, desired_length) for seg in segs]
    output_path = f"full_dataset_{patient_id}.h5"
    save_to_hdf5(fixed_segs, labs, output_path)
    print(f"已保存 {patient_id} 的数据到 {output_path}")

# # 合并结果并保存
# all_segments = []
# all_labels = []
# for segs, labs in results:
#     all_segments.extend(segs)
#     all_labels.extend(labs)
# save_to_hdf5(all_segments, all_labels, "full_dataset.h5")


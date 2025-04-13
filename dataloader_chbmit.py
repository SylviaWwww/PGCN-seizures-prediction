import mne
import numpy as np
import os
from tqdm import tqdm
import h5py


def load_edf(edf_path):
    """
    读取EDF文件，并确保所有通道均被加载
    同时如果采样率过高，则重采样到256Hz
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(raw.ch_names)
    if raw.info['sfreq'] > 256:
        raw.resample(256)
    data = raw.get_data()
    times = raw.times
    return data, times, raw.info


def parse_summary(summary_path, target_edf):
    """
    解析summary文件，返回指定EDF文件的发作时间列表
    """
    seizure_times = []
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("File Name:") and target_edf in line:
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


def extract_segments(edf_path, summary_path, segment_length=30, preictal_total_window=1800,
                     interictal_total_window=1800):
    """
    对于每个 EDF 文件，从发作前及间期切出多个30秒片段。

    - 对于发作前期（preictal）：
      对于每个发作事件，取 [seizure_start - preictal_total_window, seizure_start] 区间，
      按照非重叠的 30 秒窗口进行切割，并标记为1。

    - 对于间期（interictal）：
      如果文件中没有发作事件，则从信号开始取 [0, interictal_total_window]，
      按照 30 秒窗口切割，标记为0。
    """
    data, times, info = load_edf(edf_path)
    if data is None:
        return [], []
    fs = int(info['sfreq'])
    seizure_times = parse_summary(summary_path, os.path.basename(edf_path))

    segments = []
    labels = []

    # 提取发作前期片段（Preictal）
    for seizure_event in seizure_times:
        seizure_start, _ = seizure_event  # 以发作开始时刻作为预警期末尾
        preictal_start_time = max(0, seizure_start - preictal_total_window)
        preictal_end_time = seizure_start
        current_start = preictal_start_time
        while current_start + segment_length <= preictal_end_time:
            start_idx = int(current_start * fs)
            end_idx = int((current_start + segment_length) * fs)
            seg_data = data[:, start_idx:end_idx]
            # 若得到的片段恰好为30秒，才加入（否则跳过不完整片段）
            if seg_data.shape[1] == segment_length * fs:
                segments.append(seg_data)
                labels.append(1)  # 标记为发作前期
            current_start += segment_length  # 非重叠切分；若想有重叠可改为 current_start += overlap_step

    # 提取间期片段（Interictal）：仅当文件中没有发作事件时进行
    if not seizure_times:
        total_duration = times[-1]
        max_interictal_time = min(interictal_total_window, total_duration)
        current_start = 0
        while current_start + segment_length <= max_interictal_time:
            start_idx = int(current_start * fs)
            end_idx = int((current_start + segment_length) * fs)
            seg_data = data[:, start_idx:end_idx]
            if seg_data.shape[1] == segment_length * fs:
                segments.append(seg_data)
                labels.append(0)
            current_start += segment_length

    return segments, labels


def get_patient_channel_info(summary_path):
    """
    从 summary 文件中提取电极名称列表，并利用 standard_1020 获取对应二维坐标（取 x, y）。
    如果存在 "Channels changed:" 部分，则优先使用该部分的通道列表。
    """
    ch_names = []
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    changed_index = None
    for i, line in enumerate(lines):
        if "Channels changed:" in line:
            changed_index = i
            break
    if changed_index is not None:
        for line in lines[changed_index + 1:]:
            line = line.strip()
            if line == "" or line.startswith("File Name:"):
                break
            if line.startswith("Channel"):
                parts = line.split(":")
                if len(parts) >= 2:
                    ch = parts[1].strip()
                    if ch != "-":
                        ch_names.append(ch)
    else:
        start_index = None
        for i, line in enumerate(lines):
            if "Channels in EDF Files:" in line:
                start_index = i
                break
        if start_index is not None:
            for line in lines[start_index:]:
                line = line.strip()
                if line.startswith("Channel"):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        ch = parts[1].strip()
                        if ch != "-":
                            ch_names.append(ch)
                elif line.startswith("File Name:"):
                    break
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos = montage.get_positions()['ch_pos']
    coordinates = []
    for ch in ch_names:
        if ch in ch_pos:
            coordinates.append(ch_pos[ch][:2])
        else:
            coordinates.append([0.0, 0.0])
    coordinates = np.array(coordinates)
    return ch_names, coordinates


def save_to_hdf5(all_segments, all_labels, output_path, coordinates=None, ch_names=None):
    """
    保存数据为 HDF5 文件，同时写入 coordinates 数据集及将 ch_names 作为属性
    """
    with h5py.File(output_path, 'w') as hf:
        # 获取单个片段的形状：(n_channels, segment_length_in_samples)
        seg_shape = all_segments[0].shape
        max_shape = (None, seg_shape[0], seg_shape[1])
        hf.create_dataset('data', shape=(0, *seg_shape), maxshape=max_shape, chunks=True)
        hf.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=np.int32)
        for seg, lab in tqdm(zip(all_segments, all_labels), desc="保存数据", total=len(all_segments)):
            hf['data'].resize((hf['data'].shape[0] + 1), axis=0)
            hf['data'][-1] = seg
            hf['labels'].resize((hf['labels'].shape[0] + 1), axis=0)
            hf['labels'][-1] = lab
        if coordinates is not None:
            hf.create_dataset("coordinates", data=coordinates)
        if ch_names is not None:
            hf.attrs["ch_names"] = np.array(ch_names, dtype='S')


def pad_or_truncate(seg, desired_length):
    """
    如果 EEG 片段 seg 的采样点数大于 desired_length，则截断；
    如果小于 desired_length，则在后面用0填充；
    如果等于，则直接返回。
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


def pad_channels(seg, desired_channels):
    """
    对 EEG 片段 seg 进行通道数调整：
    - 如果实际通道数小于 desired_channels，则在下方补零；
    - 如果实际通道数大于 desired_channels，则裁剪多余通道。
    """
    seg = np.array(seg)
    current_channels = seg.shape[0]
    if current_channels < desired_channels:
        pad = np.zeros((desired_channels - current_channels, seg.shape[1]), dtype=seg.dtype)
        seg = np.concatenate([seg, pad], axis=0)
    elif current_channels > desired_channels:
        seg = seg[:desired_channels, :]
    return seg


def process_patient(subject_dir, segment_length=30, preictal_total_window=1800, interictal_total_window=1800):
    """
    处理单个患者的所有 EDF 文件，返回患者ID、所有提取的片段、标签、通道名称及对应坐标。

    使用滑窗方式切分为多个30秒片段。
    """
    patient_id = os.path.basename(subject_dir)
    summary_path = os.path.join(subject_dir, f"chb{patient_id[3:]}-summary.txt")
    edf_files = [f for f in os.listdir(subject_dir) if f.endswith(".edf")]
    patient_segments = []
    patient_labels = []
    for edf_file in edf_files:
        edf_path = os.path.join(subject_dir, edf_file)
        segs, labs = extract_segments(edf_path, summary_path,
                                      segment_length=segment_length,
                                      preictal_total_window=preictal_total_window,
                                      interictal_total_window=interictal_total_window)
        patient_segments.extend(segs)
        patient_labels.extend(labs)
    ch_names, coords = get_patient_channel_info(summary_path)
    return patient_id, patient_segments, patient_labels, ch_names, coords


# 主流程：并行处理所有患者
subjects = [
    "data/CHB-MIT/chb01", "data/CHB-MIT/chb02", "data/CHB-MIT/chb03",
    "data/CHB-MIT/chb12", "data/CHB-MIT/chb15", "data/CHB-MIT/chb18"
]

# 固定参数设置
segment_length = 30  # 片段时长30秒
# 预警期及间期总长度，可根据需要调整以获得更多数据片段
preictal_total_window = 1800  # 单位秒，比如可以设为30分钟，获得更多预警期片段
interictal_total_window = 1800  # 单位秒
desired_length = segment_length * 256  # 30秒 * 256Hz

for subject in subjects:
    patient_id, segs, labs, ch_names, coords = process_patient(
        subject,
        segment_length=segment_length,
        preictal_total_window=preictal_total_window,
        interictal_total_window=interictal_total_window
    )
    print("patient_id:", patient_id)
    print("提取到的片段数量:", len(segs))
    if not segs:
        continue
    # 统一各个片段的通道数：按照 summary 中提取到的通道数量
    desired_channels = len(ch_names)
    fixed_segs = [pad_channels(seg, desired_channels) for seg in segs]
    fixed_segs = [pad_or_truncate(seg, desired_length) for seg in fixed_segs]
    output_dir = "patient_dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dataset_{patient_id}.h5")
    save_to_hdf5(fixed_segs, labs, output_path, coordinates=coords, ch_names=ch_names)

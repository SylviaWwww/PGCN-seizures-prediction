import h5py
import numpy as np
from tqdm import tqdm


def find_global_shapes(list_of_files):
    """先扫描所有文件，找出全局 max_channels, max_times."""
    max_channels = 0
    max_times = 0
    for fpath in list_of_files:
        with h5py.File(fpath, 'r') as hf:
            # data shape: (N, c, t)
            shape = hf['data'].shape
            c = shape[1]
            t = shape[2]
            if c > max_channels:
                max_channels = c
            if t > max_times:
                max_times = t
    return max_channels, max_times


def merge_hdf5_files(list_of_files, out_file):
    if len(list_of_files) == 0:
        print("没有要合并的H5文件，结束。")
        return

    # ----------- 第1步：先找出全局最大 c,t -----------
    global_max_channels, global_max_times = find_global_shapes(list_of_files)
    print("全局最大通道数 =", global_max_channels)
    print("全局最大时间长度 =", global_max_times)

    # 第一个文件用来拿到 dtype，但不创建固定通道数
    with h5py.File(list_of_files[0], 'r') as hf_first:
        dtype_data = hf_first['data'].dtype
        dtype_labels = hf_first['labels'].dtype

    # ----------- 第2步：新建目标文件，创建可扩展数据集 -----------
    with h5py.File(out_file, 'w') as hf_out:
        maxshape = (None, global_max_channels, global_max_times)  # 只让第0维可扩展
        dset_data = hf_out.create_dataset(
            'data', shape=(0, global_max_channels, global_max_times),
            maxshape=maxshape, chunks=True, dtype=dtype_data
        )
        dset_labels = hf_out.create_dataset(
            'labels', shape=(0,), maxshape=(None,), chunks=True, dtype=dtype_labels
        )

        # ----------- 第3步：逐个文件追加写入 -----------
        total_samples = 0
        for fpath in tqdm(list_of_files, desc="合并H5文件"):
            with h5py.File(fpath, 'r') as hf_in:
                data_in = hf_in['data']  # shape: [N, c, t]
                labels_in = hf_in['labels']  # shape: [N,]

                N, c, t = data_in.shape
                n_new = N

                # 生成一个新数组 tmp_data, shape=[N, global_max_channels, global_max_times]
                # 先全部用0填充，再把原数据贴进去
                tmp_data = np.zeros(
                    (N, global_max_channels, global_max_times),
                    dtype=dtype_data
                )
                # 在通道和时间两个维度上做最小裁剪
                use_c = min(c, global_max_channels)
                use_t = min(t, global_max_times)

                # 拷贝到 tmp_data 的左上角
                tmp_data[:, :use_c, :use_t] = data_in[:, :use_c, :use_t]

                # 调整目标数据集的大小
                dset_data.resize((dset_data.shape[0] + n_new), axis=0)
                dset_labels.resize((dset_labels.shape[0] + n_new), axis=0)

                # 写入
                dset_data[-n_new:] = tmp_data
                dset_labels[-n_new:] = labels_in[...]

                total_samples += n_new

        print(f"合并完成，共计 {total_samples} 条样本。输出文件: {out_file}")


if __name__ == "__main__":
    all_patient_files = [
        "single_patient_dataset/full_dataset_chb01.h5", "single_patient_dataset/full_dataset_chb02.h5",
        "single_patient_dataset/full_dataset_chb03.h5",
        "single_patient_dataset/full_dataset_chb04.h5", "single_patient_dataset/full_dataset_chb05.h5",
        "single_patient_dataset/full_dataset_chb06.h5",
        "single_patient_dataset/full_dataset_chb07.h5", "single_patient_dataset/full_dataset_chb08.h5",
        "single_patient_dataset/full_dataset_chb09.h5",
        "single_patient_dataset/full_dataset_chb10.h5", "single_patient_dataset/full_dataset_chb11.h5",
        "single_patient_dataset/full_dataset_chb12.h5",
        "single_patient_dataset/full_dataset_chb13.h5", "single_patient_dataset/full_dataset_chb14.h5",
        "single_patient_dataset/full_dataset_chb15.h5",
        "single_patient_dataset/full_dataset_chb16.h5", "single_patient_dataset/full_dataset_chb17.h5",
        "single_patient_dataset/full_dataset_chb18.h5",
        "single_patient_dataset/full_dataset_chb19.h5", "single_patient_dataset/full_dataset_chb20.h5",
        "single_patient_dataset/full_dataset_chb21.h5",
        "single_patient_dataset/full_dataset_chb22.h5", "single_patient_dataset/full_dataset_chb23.h5",
        "single_patient_dataset/full_dataset_chb24.h5",

    ]

    merge_hdf5_files(all_patient_files, "all_patients_merged.h5")

import numpy as np

def extract_object_data(obj_data):
    """
    提取object类型标量内部的实际数据
    :param obj_data: 从npy读取的object类型标量（形状为()）
    :return: 转换后的numpy数组（若失败则返回原数据）
    """
    try:
        # 提取标量object内部的实际值
        inner_data = obj_data.item()
        # 转换为numpy数组（兼容列表/数组等类型）
        if isinstance(inner_data, (list, np.ndarray)):
            return np.array(inner_data)
        else:
            return np.array([inner_data])  # 单个值转为数组
    except Exception as e:
        print(f"⚠️ 提取内部数据时警告：{e}，使用原始数据")
        return obj_data

def compare_npy_files(file1_path, file2_path):
    """
    对比两个.npy文件的差异（完美支持object类型嵌套数组）
    :param file1_path: 第一个npy文件路径
    :param file2_path: 第二个npy文件路径
    """
    # ===================== 1. 读取文件 =====================
    try:
        data1 = np.load(file1_path, allow_pickle=True)
        data2 = np.load(file2_path, allow_pickle=True)
        print("✅ 成功读取两个npy文件（已开启pickle支持）")
    except FileNotFoundError as e:
        print(f"❌ 文件不存在：{e}")
        return
    except Exception as e:
        print(f"❌ 读取文件出错：{e}")
        return

    # ===================== 2. 基本信息对比 =====================
    print("\n===== 基本信息对比 =====")
    print(f"文件1形状：{data1.shape} | 数据类型：{data1.dtype}")
    print(f"文件2形状：{data2.shape} | 数据类型：{data2.dtype}")
    
    # 检查外层形状是否一致
    if data1.shape != data2.shape:
        print("❌ 两个文件的外层形状（维度）不一致")
        return
    else:
        print("✅ 两个文件的外层形状（维度）一致")

    # ===================== 3. 处理Object类型数据 =====================
    # 如果是object类型且形状为()，提取内部数据
    if data1.dtype == np.object_ and data1.shape == ():
        data1 = extract_object_data(data1)
    if data2.dtype == np.object_ and data2.shape == ():
        data2 = extract_object_data(data2)
    
    # 重新检查处理后的形状
    print(f"\n处理后 - 文件1形状：{data1.shape} | 数据类型：{data1.dtype}")
    print(f"处理后 - 文件2形状：{data2.shape} | 数据类型：{data2.dtype}")
    
    if data1.shape != data2.shape:
        print("❌ 处理后的数据形状不一致，无法对比内容")
        return

    # ===================== 4. 数值/内容一致性对比 =====================
    print("\n===== 内容一致性对比 =====")
    try:
        # 区分数值型和非数值型
        if np.issubdtype(data1.dtype, np.number) and np.issubdtype(data2.dtype, np.number):
            # 数值型：带浮点容差对比
            is_identical = np.allclose(data1, data2, atol=1e-9)
            if is_identical:
                print("✅ 两个文件的数值完全相同（浮点容差1e-9）")
                return
            else:
                print("❌ 两个文件的数值存在差异")
                
                # ===================== 5. 数值差异统计 =====================
                print("\n===== 差异统计分析 =====")
                diff = data1 - data2
                diff_indices = np.where(np.abs(diff) > 1e-9)
                
                print(f"差异元素总数：{len(diff_indices[0])}")
                print(f"差异最大值：{np.max(np.abs(diff)):.8f}")
                print(f"差异最小值：{np.min(np.abs(diff)):.8f}")
                print(f"差异均值：{np.mean(np.abs(diff)):.8f}")
                print(f"差异标准差：{np.std(np.abs(diff)):.8f}")

                # ===================== 6. 输出前10个差异位置 =====================
                print("\n===== 前10个差异位置及数值（示例） =====")
                if len(diff_indices[0]) > 0:
                    for i in range(min(10, len(diff_indices[0]))):
                        idx = tuple([dim[i] for dim in diff_indices])
                        val1 = data1[idx]
                        val2 = data2[idx]
                        diff_val = diff[idx]
                        print(f"位置{idx}：文件1={val1:.8f} | 文件2={val2:.8f} | 差值={diff_val:.8f}")
        else:
            # 非数值型：逐元素严格对比
            print("⚠️ 检测到非数值型数据，进行逐元素严格对比")
            # 先转为列表方便逐元素对比（兼容所有类型）
            list1 = data1.tolist() if isinstance(data1, np.ndarray) else data1
            list2 = data2.tolist() if isinstance(data2, np.ndarray) else data2
            
            if list1 == list2:
                print("✅ 两个文件的内容完全相同")
            else:
                print("❌ 两个文件的内容存在差异（非数值型无法统计差值）")
                # 尝试输出前10个差异位置（仅一维数组）
                if isinstance(list1, list) and isinstance(list2, list) and len(list1) == len(list2):
                    print("\n===== 前10个差异位置（示例） =====")
                    diff_count = 0
                    for i in range(len(list1)):
                        if list1[i] != list2[i]:
                            print(f"位置[{i}]：文件1={list1[i]} | 文件2={list2[i]}")
                            diff_count += 1
                            if diff_count >= 10:
                                break
    except Exception as e:
        print(f"⚠️ 对比时出现警告：{e}")
        print("无法完成自动对比，建议手动检查数据内容：")
        print(f"文件1前5个元素：{data1[:] if hasattr(data1, '__getitem__') else data1}")
        print(f"文件2前5个元素：{data2[:] if hasattr(data2, '__getitem__') else data2}")

# ===================== 执行对比 =====================
if __name__ == "__main__":
    # 替换为你的文件实际路径
    file1 = "res/svmguide31.0Clipmu1e-4_version_1222.npy"
    file2 = "res/svmguide31.0Clipmu1e-4.npy"
    compare_npy_files(file1, file2)
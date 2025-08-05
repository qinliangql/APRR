# 目标文件夹
target_folder="/aiarena/gpfs/code/code/OVMM/home-robot/data/hssd-hab/objects/d/"
# 下载文件的基URL
base_url="https://huggingface.co/datasets/hssd/hssd-hab/resolve/ovmm/objects/d"

# 遍历目标文件夹，查找所有 .glb 文件
for file in "$target_folder"/*.ply
do
    # 获取文件名（不包括路径）
    filename=$(basename "$file")
    
    # 构造完整的下载URL
    url="$base_url/$filename"
    
    # 下载文件到指定文件夹
    wget -P "/aiarena/gpfs/code/code/OVMM/home-robot/data/cache/d" "$url"
    
    # 打印下载状态
    if [ $? -eq 0 ]; then
        echo "Downloaded $filename successfully."
    else
        echo "Failed to download $filename."
    fi
done
# wget -P "/aiarena/gpfs/code/code/OVMM/home-robot/data/cache/e" "https://huggingface.co/datasets/hssd/hssd-hab/resolve/ovmm/objects/e/ebcbc9105cb08694adb8dd6f7a154ac70774da7f.filteredSupportSurface.ply"
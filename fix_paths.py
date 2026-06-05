import os

base_dir = r"D:\APIIT\FYP\Traffic-Sign-Recognition\sri_lankan_traffc_signs\synthetic"
splits_dir = os.path.join(base_dir, "splits")

for split_file in ["train.txt", "val.txt", "test.txt"]:
    file_path = os.path.join(splits_dir, split_file)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # If it's already an absolute path, skip
                if not os.path.isabs(line):
                    # It's a relative path like "images/syn_0001.jpg"
                    new_lines.append(os.path.join(base_dir, line) + "\n")
                else:
                    new_lines.append(line + "\n")
                    
        with open(file_path, "w") as f:
            f.writelines(new_lines)
        print(f"Fixed paths in {split_file}")

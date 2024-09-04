import os

# กำหนดโฟลเดอร์ที่มีไฟล์ภาพที่คุณต้องการเปลี่ยนชื่อ
folder_path = 'D:/Python/Special Topics in Data Analytic II/Images'

# ฟังก์ชันเปลี่ยนชื่อไฟล์ให้เป็นเลขที่เรียงลำดับ
def rename_files_in_folder(folder_path, prefix='images', extension='.JPG'):
    files = os.listdir(folder_path)
    files.sort()  # เรียงลำดับไฟล์ก่อนเปลี่ยนชื่อ
    for i, filename in enumerate(files):
        if filename.endswith(extension):
            new_name = f"{prefix}{i+1}{extension}"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

# เรียกใช้ฟังก์ชันเพื่อเปลี่ยนชื่อไฟล์ในโฟลเดอร์
rename_files_in_folder(folder_path, prefix='images', extension='.JPG')
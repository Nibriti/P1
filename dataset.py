import zipfile
import os
import shutil

# === Configuration ===
zip_path = zip_path =zip_path = r"C:\Users\ACER\Downloads\archive.zip"   # Corrected path
  # <-- Change to your ZIP file path
extract_to = "asl_alphabet_train"                # Folder where ZIP contents are extracted
labels = ['A', 'B', 'C', 'D']          # Your label folders

# === Step 1: Extract ZIP ===
if not os.path.exists(extract_to):
    os.makedirs(extract_to)

print(f"Extracting '{zip_path}' to '{extract_to}'...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
print("Extraction complete.")

# === Step 2: Create label folders if they don't exist ===
for label in labels:
    label_folder = os.path.join(extract_to, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
        print(f"Created folder: {label_folder}")

# === Step 3: Reorganize files by prefix ===
print("Reorganizing files into label folders...")
for filename in os.listdir(extract_to):
    filepath = os.path.join(extract_to, filename)
    if os.path.isfile(filepath):
        for label in labels:
            prefix = label + "_"
            if filename.startswith(prefix):
                dest_path = os.path.join(extract_to, label, filename)
                shutil.move(filepath, dest_path)
                print(f"Moved {filename} -> {label}/")
                break

print("Reorganization complete.")

import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "chaitanyakck/medical-text", output_dir="/c20250202/tianleniu/data/Medical_Text"
)

print("Path to dataset files:", path)

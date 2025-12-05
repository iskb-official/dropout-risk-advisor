# Create this Python script to organize your files
# save_as: organize_deployment.py

import shutil
from pathlib import Path
import os

# Your current path
source_path = Path(r"C:\Users\shaki\CCNU\OneDrive - mails.ccnu.edu.cn\Desktop\EDXAI")
csv_file = "students_dropout_academic_success.csv"

# Create deployment folder
deploy_path = Path(r"C:\Users\shaki\CCNU\OneDrive - mails.ccnu.edu.cn\Desktop\EDXAI-Deployment")
deploy_path.mkdir(exist_ok=True)

# Create .streamlit folder
(deploy_path / ".streamlit").mkdir(exist_ok=True)

print(f"✅ Created deployment folder at: {deploy_path}")
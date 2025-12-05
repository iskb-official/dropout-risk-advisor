import pandas as pd
from pathlib import Path

path = Path(r"C:\Users\shaki\CCNU\OneDrive - mails.ccnu.edu.cn\Desktop\EDXAI")
df = pd.read_csv(path / "students_dropout_academic_success.csv")

print(df.shape)
print(df.head())
print(df.columns)
for col in df.columns:
    print(col, df[col].unique()[:10])

import pandas as pd
import numpy as np

np.random.seed(42)

n = 500  # real ML size

df = pd.DataFrame({
    "Age": np.random.randint(18, 60, n),
    "DailyRate": np.random.randint(200, 1500, n),
    "DistanceFromHome": np.random.randint(1, 30, n),
    "MonthlyIncome": np.random.randint(2000, 20000, n),
    "YearsAtCompany": np.random.randint(0, 20, n),
    "Department": np.random.choice(["Sales", "HR", "R&D"], n),
    "OverTime": np.random.choice(["Yes", "No"], n),
})

# realistic rule-based target (Attrition)
df["Attrition"] = np.where(
    (df["MonthlyIncome"] < 5000) &
    (df["OverTime"] == "Yes") &
    (df["YearsAtCompany"] < 3),
    "Yes",
    "No"
)

df.to_csv("data/HR_data.csv", index=False)

print("Dataset generated with shape:", df.shape)
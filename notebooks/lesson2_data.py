"""Lesson 2 (throwaway): load each preprocessed HDFS file and inspect it."""
import numpy as np
import pandas as pd

D = "data/preprocessed/"
pd.set_option("display.width", 160, "display.max_columns", 40, "display.max_colwidth", 60)

def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

# 1. Templates
section("HDFS.log_templates.csv")
tpl = pd.read_csv(D + "HDFS.log_templates.csv")
print("shape:", tpl.shape)
print(tpl.dtypes)

# 2. Labels
section("anomaly_label.csv")
lab = pd.read_csv(D + "anomaly_label.csv")
print("shape:", lab.shape)
print(lab.dtypes)
print("unique BlockIds:", lab["BlockId"].nunique())
vc = lab["Label"].value_counts()
print(pd.DataFrame({"count": vc, "percent": (vc / len(lab) * 100).round(3)}))
print("normal : anomaly ratio = %.1f : 1" % (vc["Normal"] / vc["Anomaly"]))

# 3. Event traces (sequences)
section("Event_traces.csv")
tr = pd.read_csv(D + "Event_traces.csv")
print("shape:", tr.shape)
print(tr.dtypes)
print("Label values:\n", tr["Label"].value_counts())
print("Type values (top 10, incl. NaN):\n", tr["Type"].value_counts(dropna=False).head(10))
print("Type is NaN for Success rows:", tr.loc[tr.Label == "Success", "Type"].isna().all())
print("\nFeatures example:", tr.loc[0, "Features"][:80], "...")
print("TimeInterval example:", str(tr.loc[0, "TimeInterval"])[:80], "...")
print("Latency example:", tr.loc[0, "Latency"])
seq_len = tr["Features"].str.strip("[]").str.split(",").str.len()
print("\nSequence length stats:\n", seq_len.describe().round(1))

# 4. Event occurrence matrix (counts)
section("Event_occurrence_matrix.csv")
mx = pd.read_csv(D + "Event_occurrence_matrix.csv")
print("shape:", mx.shape)
print("columns:", list(mx.columns))
print(mx.head(3))
ecols = [f"E{i}" for i in range(1, 30)]
print("row sum of counts == sequence length for all rows:",
      bool((mx[ecols].sum(axis=1).values == seq_len.values).all()))

# 5. Do the files agree?
section("Consistency checks")
print("Same BlockId order (labels vs traces):", bool((lab.BlockId.values == tr.BlockId.values).all()))
print("Same BlockId order (traces vs matrix):", bool((tr.BlockId.values == mx.BlockId.values).all()))
m = lab.merge(tr[["BlockId", "Label"]], on="BlockId", suffixes=("_lab", "_tr"))
print(pd.crosstab(m.Label_lab, m.Label_tr))

# 6. Bonus file the brief didn't mention
section("HDFS.npz")
z = np.load(D + "HDFS.npz", allow_pickle=True)
for k in z.files:
    print(k, z[k].shape, z[k].dtype)

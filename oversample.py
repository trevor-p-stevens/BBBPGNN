# %% Load Dataset
import pandas as pd

data = pd.read_csv("data/raw/BBBP.csv")

data.index = data["num"]
data["name"].value_counts()
start = data.iloc[1]["num"]
#print(start)

# %% Oversample

neg_n = data["p_np"].value_counts()[1]
pos_n = data["p_np"].value_counts()[0]
print(pos_n/neg_n)
mult = int(neg_n/pos_n) - 1

print(mult)

repl = [data[data["p_np"] == 0]]*mult

df_repl = pd.DataFrame(repl[0], columns=['num','name','p_np','smiles'])

data = pd.concat([data,df_repl], ignore_index=True)

print(data.shape)
data = data.sample(frac=1).reset_index(drop=True)

index = range(start, start + data.shape[0])
data.index = index
data["num"] = data.index
data.head()

# %% Save

data.to_csv("data/raw/BBBP_train_over.csv")

# %%

#%%

import pandas as pd
import json
import re
import matplotlib.pyplot as plt

df = pd.read_csv("navne.csv", sep=";")
justnames = list(df['Navn'])

# Filter non-ascii characters, keeping åæø
def replace_chars(name: str) -> str:
    name = name.lower()
    name = name.replace("ø", "o")
    name = name.replace("å", "a")
    name = name.replace("æ", "ae")
    name = re.sub(r'[^\x00-\x7f]',r'', name) # ensure only ascii characters remain
    return name

def filter_names(name: str):
    if "-" in name or "'" in name:
        return False
    return True

fnames = list(filter(filter_names, map(replace_chars, justnames)))

#%%

# plot name length distribution
namelendist = [len(x) for x in fnames]
maxnamelen = max(namelendist)
plt.hist(namelendist, bins=range(1,maxnamelen, 1))
plt.show()

# %%

# few names are dropped cutting from 17 to 10, worth cutting the complexity
print(len(fnames))
names = [n for n in fnames if len(n) <= 10]
print(len(names))
maxnamelen = max(namelendist)

# %%

# Save jsons
# itoc and ctoi technically easier to do with ord() and chr() in this case
with open("names.json", "w") as f:
    json.dump(names, f, indent=2)

letters = sorted(list(set("".join(names))))
i_to_c = {i:l for (i,l) in enumerate(letters)}
c_to_i = {l:i for (i,l) in i_to_c.items()}
with open("i_to_c.json", "w") as f:
    json.dump(i_to_c, f, indent=2)
with open("c_to_i.json", "w") as f:
    json.dump(c_to_i, f, indent=2)



# %%
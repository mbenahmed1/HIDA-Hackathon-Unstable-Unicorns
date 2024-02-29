import pandas as pd

models = ['effunet_small', 'unet_small']
augmentation = [True]
in_channels = [2, 3, 5]

df = pd.DataFrame()
idx = 0
for model in models:
    for aug in augmentation:
        for in_channel in in_channels:
            df.loc[idx, "model"] = model
            df.loc[idx, "augmentation"] = aug
            df.loc[idx, "in_channels"] = in_channel
            df.loc[idx, "batch_size"] = 2
            idx += 1
            
df["in_channels"] = df["in_channels"].astype(int)
df["batch_size"] = df["batch_size"].astype(int)
df.to_csv("combinations_train_monai_2.csv", index=False)

import re
import pandas as pd
import matplotlib.pyplot as plt

def read_values(path, n_exp=5, n_active_iter=10, n_epoch=20, n_classes=10,active=True):
    with open(path,mode='r') as log_file:
        lines = log_file.readlines()
        pattern = re.compile("^(.*?)Sub epoch (\d+) train acc: (0.\d{3}) train loss: (\d{1,3}.\d{4}) train f1: (\d.\d{4}) val acc: (0.\d{3}) val loss: (\d{1,3}.\d{4}) val f1: (\d.\d{4}) test acc: (0.\d{3}) test loss: (\d{1,3}.\d{4}) test f1: (\d.\d{4})|^(.*?)Weighted f1: (0.\d{4})")
        pattern_perclass = re.compile("^(.*?)Class: ([a-zA-Z\s]+)(\s)*test accuracy: (0.\d{3}) test f1: (0.\d{4})")
        values, values_perclass = [], []
        for line in lines:
            m = pattern.match(line)
            if m:
                last_epoch = m.groups()[1]
                # print(m.groups()[1:])
                values.append(m.groups()[1:])
            m = pattern_perclass.match(line)
            if m:
                values_perclass.append([last_epoch]+ [m.groups()[1]] + list(m.groups()[-2:]))
        columns = ["epoch", "train acc", "train loss", "train f1", "val acc", "val loss", "val f1", "test acc", "test loss", "test f1","timestamp", "weighted f1"]
        df = pd.DataFrame(values, columns=columns)
        df["weighted f1"] = df["weighted f1"].bfill()
        df = df.loc[~df.epoch.isna()].drop(columns="timestamp")
        df = df.apply(pd.to_numeric, errors='ignore')
        if active:
            df["active_iter"] = [i // n_epoch for i in range(n_epoch*n_active_iter)]*n_exp
        df["experiment"] = [j for j in range(5) for _ in range(n_epoch*n_active_iter)]
        
        # df["experiment"] = [j for j in range(5) for _ in range(20)]
        
        df_perclass = pd.DataFrame(values_perclass, columns=["epoch","class", "test acc", "test f1"])
        df_perclass = df_perclass.apply(pd.to_numeric, errors ='ignore')
        if active:
            df_perclass["active_iter"] = [i // (n_epoch*n_classes) for i in range(n_epoch*n_active_iter*n_classes)]*n_exp
        df_perclass["experiment"] = [j for j in range(5) for _ in range(n_epoch*n_active_iter*n_classes)]
        return df, df_perclass

def create_plots(df, exp_name):
    df = df.set_index(["experiment","epoch"])
    
    for m in ["acc", "loss", "f1"]:
        _, ax = plt.subplots(figsize=(8,6))
        df.groupby("experiment").plot(y=[f"val {m}", f"test {m}"], ax=ax)
        plt.savefig(f"imgs/{exp_name}_{m}.jpg")
        
        _, ax = plt.subplots(figsize=(8,6))
        df.groupby("epoch").mean().plot(y=[f"val {m}", f"test {m}"], ax=ax)
        plt.savefig(f"imgs/{exp_name}_{m}_avg.jpg")
    
    _, ax = plt.subplots(figsize=(8,6))
    df.groupby("epoch").mean().plot(y=["test f1", "weighted f1"], ax=ax)
    plt.savefig(f"imgs/{exp_name}_f1_scores_avg.jpg")
    
def create_perclass_plots(df, exp_name):
    df = df.set_index(["experiment","epoch", "class"])
    
    for m in ["acc", "f1"]:
        _, ax = plt.subplots(figsize=(8,6))
        df.groupby(["experiment", "class"]).plot(y=[f"test {m}"], ax=ax)
        plt.savefig(f"imgs/{exp_name}_{m}_perclass.jpg")
        
        _, ax = plt.subplots(figsize=(8,6))
        df.groupby(["epoch", "class"]).mean().plot(y=[f"test {m}"], ax=ax)
        plt.savefig(f"imgs/{exp_name}_{m}_avg_perclass.jpg")

def stack_results(file_paths: list, n_exp=5, n_active_iter=10, n_epoch=20, n_classes=10,active=True):
    dfs, dfs_perclass = [], []
    for f in file_paths:
        heuristic = "_".join(f.replace(".log","").split("/")[-1].split("_")[2:])
        df, df_perclass = read_values(f, n_exp, n_active_iter, n_epoch, n_classes, active)
        df["heuristic"] = heuristic
        df_perclass["heuristic"] = heuristic
        dfs.append(df)
        dfs_perclass.append(df_perclass)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df_perclass = pd.concat(dfs_perclass, ignore_index=True)
    final_df.to_csv(file_paths[0].replace("_None.log", ".csv"), index=False)
    final_df_perclass.to_csv(file_paths[0].replace("_None.log", "_perclass.csv"), index=False)


def save_base(file_path: str, n_exp=5, n_active_iter=1, n_epoch=20, n_classes=10, active=False):
    df, df_perclass = read_values(file_path,n_exp, n_active_iter, n_epoch, n_classes,active)
    df.to_csv(file_path.replace(".log", "_base.csv"), index=False)
    df_perclass.to_csv(file_path.replace(".log", "_base_perclass.csv"), index=False)
    
if __name__ == "__main__":
    # df, df_perclass = read_values('logs/EFFNETV2S_CIFAR10_cleaned.log')
    # create_plots(df, "EFFNETV2S_CIFAR10")
    # create_perclass_plots(df_perclass, "EFFNETV2S_CIFAR10")
    # df, df_perclass = read_values('logs/EFFNETV2S_CIFAR10_representative_sampling.log')
    # df.to_csv('logs/EFFNETV2S_CIFAR10_representative_sampling.csv', index=False)
    # df_perclass.to_csv('logs/EFFNETV2S_CIFAR10_representative_sampling_perclass.csv', index=False)
    
    # df, df_perclass = read_values('logs/EFFNETV2S_CIFAR10_None.log')
    # df.to_csv('logs/EFFNETV2S_CIFAR10_None.csv', index=False)
    # df_perclass.to_csv('logs/EFFNETV2S_CIFAR10_None_perclass.csv', index=False)
    
    # df, df_perclass = read_values('logs/EFFNETV2S_CIFAR10_largest_margin_extended.log')
    # df.to_csv('logs/EFFNETV2S_CIFAR10_largest_margin_extended.csv', index=False)
    # df_perclass.to_csv('logs/EFFNETV2S_CIFAR10_largest_margin_extended_perclass.csv', index=False)
    
    # df, df_perclass = read_values('logs/EFFNETV2S_CIFAR10_representative_mc_dropout_extended.log')
    # df.to_csv('logs/EFFNETV2S_CIFAR10_representative_mc_dropout_extended.csv', index=False)
    # df_perclass.to_csv('logs/EFFNETV2S_CIFAR10_representative_mc_dropout_extended.csv', index=False)
    
   
    # save_base('logs/OWN_FastFoodV2.log', n_epoch=10)
    # files = [
    #     "logs/OWN_FastFoodV2_None.log",
    #     "logs/OWN_FastFoodV2_largest_margin.log",
    #     "logs/OWN_FastFoodV2_smallest_margin.log",
    #     "logs/OWN_FastFoodV2_least_confidence.log",
    #     "logs/OWN_FastFoodV2_representative_mc_dropout.log",
    #     "logs/OWN_FastFoodV2_representative_sampling.log",
    #     "logs/OWN_FastFoodV2_mc_dropout.log"       
           
    # ]
    
    # stack_results(files, n_epoch=10)
    
    # 
    # save_base('logs/EFFNETV2S_PlantVillage.log', n_epoch=30)
    
    files = [
        "logs/EFFNETV2S_PlantVillage_None.log",
        "logs/EFFNETV2S_PlantVillage_largest_margin.log",
        # "logs/OWN_FastFoodV2_smallest_margin.log",
        # "logs/OWN_FastFoodV2_least_confidence.log",
        # "logs/OWN_FastFoodV2_representative_mc_dropout.log",
        # "logs/OWN_FastFoodV2_representative_sampling.log",
        "logs/EFFNETV2S_PlantVillage_mc_dropout.log"       
           
    ]

    
    stack_results(files, n_epoch=15)
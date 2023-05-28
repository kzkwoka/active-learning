import re
import pandas as pd
import matplotlib.pyplot as plt

def preprocess(path):
    with open(path,mode='r') as log_file:
        lines = log_file.readlines()
        pattern = re.compile("^(.*?)Sub epoch (\d+) train acc: (\d{1,2}.\d{2}) train loss: (\d.\d{4}) val acc: (\d{1,2}.\d{2}) val loss: (\d.\d{4}) test acc: (\d{1,2}.\d{2}) test loss: (\d.\d{4})")
        values = []
        for line in lines:
            m = pattern.match(line)
            if m:
                values.append(m.groups()[1:])
        columns = ["sub epoch", "train acc", "train loss", "val acc", "val loss", "test acc", "test loss"]
        df = pd.DataFrame(values, columns=columns)
        df = df.apply(pd.to_numeric)
        df["epoch"] = [j for j in range(5) for _ in range(30) ]
        return df              

def create_plots(df):
    df = df.set_index(["epoch","sub epoch"])
    _, ax = plt.subplots(figsize=(8,6))
    # plots = 
    df.groupby("epoch").plot(y=["val acc", "test acc"], ax=ax)
    # for idx, p in enumerate(plots):
    #     p.figure.savefig(f"acc{idx}.jpg")
    plt.savefig("test_acc_explr.jpg")
    
    _, ax = plt.subplots(figsize=(8,6))
    df.groupby("sub epoch").mean().plot(y=["val acc", "test acc"], ax=ax)
    plt.savefig("test_acc_avg_explr.jpg")
    
if __name__ == "__main__":
    # df = preprocess('IV_results/vgg16_full.log')
    df = preprocess('logs/vgg16_cifar_with_explr.log')
    create_plots(df)

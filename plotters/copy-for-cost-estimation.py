import glob
import pandas as pd

if __name__ == '__main__':

    experiment = 'RetrainAfterTime-FedProx-NonIID'
    files = glob.glob(f'data/{experiment}/training_*-hospital_Hospital-0-GR_0-seed_0.csv')
    for file in files:
        df = pd.read_csv(file)
        suffix = file.split('/')[-1]
        df.to_csv(f'data/cost-estimation/retrainaftertime/{suffix}', index=False)

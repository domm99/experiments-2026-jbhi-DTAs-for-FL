import glob


if __name__ == '__main__':

    experiments = [
        'ADWINErrorHierarchicalRetrainingPolicy-FedPers-NonIID',
        'ADWINErrorHierarchicalRetrainingPolicy-Federated-NonIID'
    ]
    data_path = 'data/'

    for experiment in experiments:

        print(f'================================ Experiment {experiment} ================================')

        for i in range(10):
            print(f'----------------------------- H {i} -----------------------------')
            files = glob.glob(f'{data_path}/{experiment}/training_*-hospital_Hospital-{i}-GR_0-seed_0.csv')
            print(f'For {experiment} --- found {len(files)} files')
            print(files)
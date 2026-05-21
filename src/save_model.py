import pickle

def save_model(model, name):
    with open(f'../artifacts/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

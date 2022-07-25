import faiss
import numpy as np

import timeit

def load_features(file):
    i
    features = np.load(file)['data'].astype(np.float32)
    features = features / np.sqrt(np.sum(np.square(features), axis=1, keepdims=True))
    print(f'Found {features.shape[0]} crops for {file}.')
    return features

if __name__ == '__main__':

    res = faiss.StandardGpuResources()
    for j in range(9):
        i = j+1
        print(f'Starting loop {i}/10')
        start = timeit.default_timer()

        # load in real data split i
        print('Loading in real features...')
        features_nn = load_features(f'splits/real_split_{i}.npz')

        # load in CARLA data part 1
        print('Loading in first set of CARLA features...')
        features_ref = load_features('splits/crop_CARLA_1.npz')
        dim = features_ref.shape[1]

        # create index of real data
        print('Setting up index...')
        nn_index_cpu = faiss.IndexFlatL2(dim)
        nn_index = faiss.index_cpu_to_gpu(res, 0, nn_index_cpu)
        nn_index.add(features_nn)

        # search
        print('Performing first search...')
        D1, I1 = nn_index.search(features_ref, 10)

        del features_ref

        # load in CARLA data part 2
        print('Loading in second set of CARLA features...')
        features_ref = load_features('splits/crop_CARLA_2.npz')
        
        D2, I2 = nn_index.search(features_ref, 10)

        D = np.append(D1, D2, axis=0)
        I = np.append(I1, I2, axis=0)

        del features_ref, nn_index, features_nn, D1, D2, I1, I2, dim

        print(f'Saving to faiss/knn_{i}.npz')
        np.savez_compressed(f'faiss/knn_{i}.npz', ind=I, dist=D)

        stop = timeit.default_timer()

        print(f'Time (Split {i}): {stop - start}') 
        del D, I

    pass

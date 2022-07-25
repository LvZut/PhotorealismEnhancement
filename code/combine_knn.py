import numpy as np
import timeit
from tqdm import tqdm
import os

def merge_row(row1, ind1, row2, ind2, index_start=0):
    done = False
    while not done:
        highest = np.amax(row1)
        lowest = np.amin(row2)

        if highest > lowest:
            # find index of highest value in row1
            rep_ind1 = np.where(row1 == highest)[0][0]
            rep_ind2 = np.where(row2 == lowest)[0][0]

            # replace that value and index with values from row2
            row1[rep_ind1] = row2[rep_ind2]
            ind1[rep_ind1] = ind2[rep_ind2] + index_start

            # set old row2 value to 2 so it doesn't get replaced
            row2[rep_ind2] = 2
        else:
            # done if lowest row2 > highest row1
            done = True
    
    return row1, ind1



def merge_features(I, D, I_split, D_split, index_start):
    D_in = D
    I_in = I

    #breakpoint()
    for row in tqdm(range(I.shape[0])):
        done = False

        I_row = I[row]
        D_row = D[row]
        I_srow = I_split[row]
        D_srow = D_split[row]


        
        D_row, I_row = merge_row(D_row, I_row, D_srow, I_srow, index_start=index_start)


    print(f'D\nBefore:\n{D_in[0]}\n\nAfter:\n{D[0]}')
    print(f'I\nBefore:\n{I_in[0]}\n\nAfter:\n{I[0]}')

    return I, D


if __name__ == '__main__':

    combined_D = np.ones((5754165, 10))*2
    combined_I = np.ones((5754165, 10))
    # combined_D = np.ones((24, 10))*2
    # combined_I = np.ones((24, 10))
    # I_split = np.reshape((np.arange(240)), (24,10))
    # D_split = np.reshape((np.arange(240) * 0.01), (24,10))

    index_start = 0
    for i in range(10):
        print(f'Starting loop {i}/10')
        start = timeit.default_timer()

        knn_split = np.load(f'faiss/knn_{i}.npz')
        I_split = knn_split['ind']
        D_split = knn_split['dist']

        assert combined_I.shape[0] == I_split.shape[0]

        I, D = merge_features(combined_I, combined_D, I_split, D_split, index_start)
        stop = timeit.default_timer()
        print(f'Loop {i} took: {stop-start} Seconds')
        
        # add index equal to size of real_crops splits
        index_start += 1813179

        del I_split, D_split, knn_split

    np.savez_compressed('knn_CARLA-real.npz', ind=combined_I, dist=combined_D)
        


    pass

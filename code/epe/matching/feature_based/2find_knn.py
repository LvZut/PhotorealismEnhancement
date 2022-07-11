from argparse import ArgumentParser
from pathlib import Path

import faiss
import numpy as np

import timeit

if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument('file_src', type=Path, help="Path to feature file for source dataset.")
    p.add_argument('file_dst', type=Path, help="Path to feature file for target dataset.")
    p.add_argument('out', type=Path, help="Path to output file with matches.")
    p.add_argument('-k', type=int, help="Number of neighbours to sample. Default = 5.", default=5)
    args = p.parse_args()

    features_ref = np.load(args.file_src)['crops'].astype(np.float32)
    
    

    # test with only 1/1000 of the starting data
    features_ref = features_ref
    features_ref = features_ref / np.sqrt(np.sum(np.square(features_ref), axis=1, keepdims=True))



    dim = features_ref.shape[1]
    print(f'Found {features_ref.shape[0]} crops for source dataset.')

    features_nn = np.load(args.file_dst)['crops'].astype(np.float32)
    features_nn = features_nn / np.sqrt(np.sum(np.square(features_nn), axis=1, keepdims=True))
    assert features_nn.shape[1] == dim
    print(f'Found {features_nn.shape[0]} crops for target dataset.')
    
    chunk_size = int(features_nn.shape[0] / 100) + 1

    for split in range(100):
        print(f'Starting split: {split}/100')
        start = timeit.default_timer()
    
        #res = faiss.StandardGpuResources()
        #res.noTempMemory()
    
        #res.setTempMemory(5000 * 1024 * 1024)
        #res.setPinnedMemory(5000 * 1024 * 1024)
    
        #res = faiss.StandardGpuResources()
        #flat_config = faiss.GpuIndexFlatConfig()
        #flat_config.useFloat16 = True
        #flat_config.device = 0
        
        #nn_index = faiss.GpuIndexFlatL2(res, dim, flat_config)
        nn_index = faiss.IndexFlatL2(dim)

        #n_chunks = int(features_nn.shape[0] / 10000) + 1
        #for i in range(n_chunks):
        print(f'Indexing {chunk_size*split} to {chunk_size*(split+1)}')
        nn_index.add(features_nn[chunk_size*split:chunk_size*(split+1)])
    
        D,I = nn_index.search(features_ref, args.k)
    
        np.savez_compressed(f'splits/{split}_{args.out}', ind=I, dist=D)
        
        
        stop = timeit.default_timer()
        print(f'Finished {split}/{100} in: {stop - start} Seconds')
    
    pass

import pickle
import numpy as np

# widths = [2, 3, 4, 8, 16, 32, 64]
batch_sizes = [8, 16, 32, 64, 96, 128]
aux_tasks = ['goaldist', 'wall0', 'wall1',  'wall01',  'none']

# for width in widths:
    # data_folder = 'data/pdistal_rim_heatmap/'
for batch in batch_sizes:
    for aux in aux_tasks:
        data_folder = 'data/pdistal_batchaux_heatmap/'
        heatmap_path = data_folder + f'nav_pdistal_batch{batch}aux{aux}_checkpoint_hms'
        heatmap_half_path = data_folder + f'nav_pdistal_batch{batch}aux{aux}_checkpoint_hms_half'
        print(heatmap_path)
        heatmaps = pickle.load(open(heatmap_path, 'rb'))

        heatmap_half = {}

        for t in heatmaps:
            heatmap_half[t] = {}
            for chk in heatmaps[t]:
                heatmap_half[t][chk] = []
                for hm in heatmaps[t][chk]:
                    heatmap_half[t][chk].append(hm.astype(np.float16))
                    
        pickle.dump(heatmap_half, open(heatmap_half_path, 'wb'))


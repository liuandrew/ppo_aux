import io
from PIL import Image

from fastai import *
from fastai.vision.all import *
from fastai.vision.widgets import *
import timm

from representation_analysis import *


def get_ep_pos_angle_from_data(res):
    '''
    Convert a res object e.g., res = all_res[batch][aux][trial][chk] to ep_pos and ep_angles
    which individual ones can be given to draw_trajectory
    '''
    ep_pos = split_by_ep(res['data']['pos'], res['dones'])
    ep_angles = split_by_ep(res['data']['angle'], res['dones'])
    ep_pos = [np.vstack(p) for p in ep_pos]
    ep_angles = [np.vstack(a) for a in ep_angles]
    
    return ep_pos, ep_angles



def draw_trajectory(pos=None, angle=None, fig=None, ax=None):
    '''Convert positions and angles into an image trajectory
    Adds a few extra details like where the start and goal reached locations are
    as well as adding increasing redness as the agent spends time in one spot rotating
    without forward movement'''
    if fig == None and ax == None:
        fig, ax = pplt.subplots()
    stopped = 0
    last_p = np.zeros(2)
    for i, (p, a) in enumerate(zip(pos, angle)):
        redness = max(0, 1-stopped*0.1)
        color = [1, redness, redness, 1]
        draw_character(p, a[0], ax=ax, color=color)

        if (p == last_p).all():
            stopped += 1
        else:
            stopped = 0
        last_p = p

    #redraw first and last steps
    draw_character(pos[0], angle[0][0], color=[0, 1, 0, 1], size=18, ax=ax)
    if len(pos) < 202:
        draw_character(pos[-1], angle[-1][0], color=[0, 1, 1, 1], size=18, ax=ax)
    ax.format(xlim=[0, 300], ylim=[0, 300])
    
    return fig, ax
    
    
def convert_trajectory_to_rgb(pos, angle, keep_rgb_dims=3, fig=None, ax=None):
    '''Given a set of positions and angles, convert trajectory into an image
    and return as an rgb array of size 224x224.
    
    Note that this requires specific rcparams from this block and plotting from proplot
    
    keep_rgb_dims: how many of the rgb dims to keep
        3: discard alpha from rgba. Only 3 dims are used for learner model'''
    fig, ax = draw_trajectory(pos, angle, fig, ax)
    #Convert to numpy array
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=85.5)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(224, 224, -1))
    io_buf.close()
    pplt.close()
    
    img_arr = img_arr[:, :, :keep_rgb_dims]
    return img_arr


def classify_res(res, batch=True):
    '''Use learner to perform classification on a specific res dictionary
    
    batch: perform inference as a batch which is 4x faster (although conversion
        to images is still  the slowest part)'''
    
    ep_pos, ep_angles = get_ep_pos_angle_from_data(res)

    if batch:
        imgs = []
        for i in range(len(ep_pos)):
            img = convert_trajectory_to_rgb(ep_pos[i], ep_angles[i])
            imgs.append(img)
        
        with learner.no_bar():
            test_dl = learner.dls.test_dl(imgs)
            probs, _ = learner.get_preds(dl=test_dl)

            idxs = np.array(probs).argmax(axis=1)
            id_to_vocab = dict(zip(range(len(learner.dls.vocab)), learner.dls.vocab))
            labels = list(map(lambda x: id_to_vocab[x], idxs))
        
    else:
        # Generate predictions for each trajectory in trial
        labels = []
        idxs = []
        probs = []

        with learner.no_bar():
            for i in range(len(ep_pos)):
                img = convert_trajectory_to_rgb(ep_pos[i], ep_angles[i])
                label, idx, prob = learner.predict(img)
                labels.append(label)
                idxs.append(idx.item())
                probs.append(prob.numpy())

    return {
        'labels': labels,
        'idxs': idxs,
        'probs': probs
    }



def classify_folder(path, learner):
    '''Use learner to perform batch classification on a directory of images
    E.g. folder = Path('data/trajectories/test_imgs/batch16_auxnone_trial0_chk900')
    '''
    with learner.no_bar():
        test_dl = learner.dls.test_dl(get_image_files(path), batch_size=100)
        probs, _ = learner.get_preds(dl=test_dl)
        
        idxs = np.array(probs).argmax(axis=1)
        id_to_vocab = dict(zip(range(len(learner.dls.vocab)), learner.dls.vocab))
        labels = list(map(lambda x: id_to_vocab[x], idxs))
        
    return {
        'labels': labels,
        'idxs': idxs,
        'probs': probs
    }


    
    


'''Trajectory image saving functions'''

img_folder = Path('data/trajectories/imgs')
'''Create folder in data folder'''
id_to_label = {
    0: 'favorite_spot',
    1: 'direct',
    2: 'near_miss',
    3: 'circling',
    4: 'test_corner',
    5: 'uncertain_direct',
    6: 'stuck'
}
for _, label in id_to_label.items():
    folder = img_folder/label
    folder.mkdir(exist_ok=True)

def save_trajectory(pos, angle, file):
    img_arr = convert_trajectory_to_rgb(pos, angle)

    im = Image.fromarray(img_arr)
    im.save(file)
    
    
    
'''
Plotting functions
'''

def generate_grid_array(n):
    '''
    Generate an array that is used by proplot subplots to create a
    grid that is as close to a square number as possible
    '''
    box_len = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / box_len))
    last_row_remainder = box_len - (n % box_len)
    left_missing = int(np.floor(last_row_remainder / 2))
    right_missing = int(np.ceil(last_row_remainder / 2))

    # How many full rows to add by using np.arange
    aranged_rows = n_rows - 1
    if last_row_remainder == box_len:
        aranged_rows += 1
    
    # Add in those rows
    array = []
    for i in range(aranged_rows):
        array.append(np.arange(i*box_len+1, ((i+1)*box_len)+1))
        last_idx = array[-1][-1]    

    # Generate a last row with symmetric missing plots if needed
    if last_row_remainder != box_len:
        last_row = np.zeros(box_len)
        for i in range(left_missing, box_len - right_missing):
            last_row[i] = last_idx + 1
            last_idx += 1
        array.append(last_row)
        
        
    array = np.vstack(array).astype('int')
    return array


def set_trajectory_plot_style(reset=False):
    '''Set up rc params for proplot so that we get nice images to feed into
    the visual system for ananlysis
    
    If reset == True, revert to default rcprams for normal plotting'''
    if reset:
        pplt.rc.reset()
    else:
        pplt.rc.update({
            'axes.spines.bottom': False,
            'axes.spines.top': False,
            'axes.spines.left': False,
            'axes.spines.right': False,
            'axes.facecolor': 'black',
            'axes.grid': False
        })
    
#Note we call set_trajectory_plot_style here
set_trajectory_plot_style()
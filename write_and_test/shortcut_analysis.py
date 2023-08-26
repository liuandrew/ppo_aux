import numpy as np
from evaluation import *

turn_speed = 0.3
move_speed = 10

shortcut_target = np.array([150, 250])
path_target = np.array([25, 250])
goal_target = np.array([275, 275])

def get_moves_to_target(start_pos, target_pos, start_angle, perfect_angle=False):
    '''
    Get the direction and turns needed to head towards a certain direction
    
    perfect_angle: if True, assume we can reach the perfect angle needed rather
        than requiring harder turns
    '''
    target_vector = target_pos - start_pos
    target_angle = np.arctan2(target_vector[1], target_vector[0])

    # Calculate the angle needed to head in vector direction to target
    #  and how many turns are needed to get there
    # theta = abs(target_angle - angle) % (2 * np.pi)
    # min_angle_dist = 2 * np.pi - theta if theta > np.pi else theta
    # direction = -1 if theta > np.pi else 1
    
    angle_dist1 = target_angle - start_angle
    angle_dist2 = min(abs(2*np.pi+angle_dist1), abs(2*np.pi-angle_dist1))
    sign1 = np.sign(angle_dist1)
    angle_dist1 = abs(angle_dist1)

    if angle_dist1 < angle_dist2:
        direction = sign1
    else:
        direction = -1 * sign1
    min_angle_dist = min(angle_dist1, angle_dist2)
    
    turns = np.round(min_angle_dist / turn_speed)
    
    if perfect_angle:
        resulting_angle = target_angle
    else:
        resulting_angle = (direction * turns * turn_speed + start_angle) % (2*np.pi)
    resulting_vector = np.array([np.cos(resulting_angle), np.sin(resulting_angle)])

    return resulting_angle, resulting_vector, turns


def compute_path_steps(pos, angle, target, ret_pos=False, perfect_angle=False, 
                      ret_steps=False):
    '''Compute a theoretical best bath through the given target
    
    ret_pos: whether to return the intermediate positions
    perfect_angle: whether to assume that we can turn perfectly to the correct angle
    ret_steps: whether to return the individual steps used'''
    
    # Find number of steps to get above y of 250 via shortcut
    angle1, dir1, turns1 = get_moves_to_target(pos, target, angle, perfect_angle)
    steps1 = np.ceil((250 - pos[1]) / (move_speed * dir1[1]))
    steps1 = steps1//2
    pos1 = steps1 * move_speed * dir1 + pos

    angle1b, dir1b, turns1b = get_moves_to_target(pos1, target, angle1, perfect_angle)
    steps1b = np.ceil((250 - pos1[1]) / (move_speed * dir1b[1]))
    pos1b = steps1b * move_speed * dir1b + pos1


    angle2, dir2, turns2 = get_moves_to_target(pos1b, goal_target, angle1b, perfect_angle)
    steps2 = np.ceil((262.5 - pos1b[0]) / (move_speed * dir2[0]))
    steps2 = steps2//2
    pos2 = steps2 * move_speed * dir2 + pos1b

    angle2b, dir2b, turns2b = get_moves_to_target(pos2, goal_target, angle2, perfect_angle)
    steps2b = np.ceil((262.5 - pos2[0]) / (move_speed * dir2b[0]))
    pos2b = steps2b * move_speed * dir2b + pos2

    total_steps = np.sum([turns1, turns1b, turns2, turns2b, steps1, steps1b, steps2, steps2b])
    
    if ret_pos:
        if ret_steps:
            return total_steps, [pos1, pos1b, pos2, pos2b], [turns1, turns1b, turns2, turns2b, steps1, steps1b, steps2, steps2b]
        return total_steps, [pos1, pos1b, pos2, pos2b]
    elif ret_steps:
        return total_steps, [turns1, turns1b, turns2, turns2b, steps1, steps1b, steps2, steps2b]
    
    return total_steps


def check_shortcut_usage(p, ret_arrays=False):
    '''
    Check if shortcut was used in a trajectory
    '''
    # Check that x values are within range
    x1 = (p[:-1, 0] > 125) & (p[:-1, 0] < 175)
    x2 = (p[1:, 0] > 125) & (p[1:, 0] < 175)

    # Check that y values cross over
    y1 = (p[:-1, 1] < 250)
    y2 = (p[1:, 1] > 250)

    # Crossed shortcut
    used_shortcut = ((y1 & y2) & (x1 | x2)).any()
    
    if ret_arrays:
        return used_shortcut, [x1, x2, y1, y2]
    
    return used_shortcut



def test_shortcut_agent(model, obs_rms, character_reset_pos=0, n_eps=20):
    env_kwargs = {'character_reset_pos': character_reset_pos, 'shortcut_probability': 1.}
    res1 = evaluate(model, obs_rms, env_kwargs=env_kwargs, env_name='ShortcutNav-v0',
                  num_episodes=n_eps, data_callback=shortcut_data_callback)

    env_kwargs = {'character_reset_pos': character_reset_pos, 'shortcut_probability': 0}
    res2 = evaluate(model, obs_rms, env_kwargs=env_kwargs, env_name='ShortcutNav-v0',
                  num_episodes=n_eps, data_callback=shortcut_data_callback)

    # Check how often shortcuts were used when available
    shortcuts_used = np.array([check_shortcut_usage(p) for p in res1['data']['pos']])

    # Check how well agent did compared to an "optimal" path
    shortcut_starts = []
    shortcut_optimal_steps = []
    for i in range(n_eps):
        # First position and angle of the episode
        p = res1['data']['pos'][i][0]
        a = res1['data']['angle'][i][0]
        shortcut_starts.append((p, a))
        shortcut_optimal_steps.append(compute_path_steps(p, a, shortcut_target, perfect_angle=True))

    path_optimal_steps = []
    path_starts = []
    for i in range(n_eps):
        p = res2['data']['pos'][i][0]
        a = res2['data']['angle'][i][0]
        path_starts.append((p, a))
        path_optimal_steps.append(compute_path_steps(p, a, path_target, perfect_angle=True))

    shortcut_actual_steps = [len(r) for r in res1['rewards']]
    path_actual_steps = [len(r) for r in res2['rewards']]
    
    return {
        'shortcuts_used': shortcuts_used,
        'shortcut_use_rate': np.sum(shortcuts_used)/n_eps,
        'shortcut_actual_steps': shortcut_actual_steps,
        'shortcut_theoretical_steps': np.array(shortcut_optimal_steps).squeeze() - 2,
        'shortcut_starts': shortcut_starts,
        'path_actual_steps': path_actual_steps,
        'path_theoretical_steps': np.array(path_optimal_steps).squeeze() - 2,
        'path_starts': path_starts
    }

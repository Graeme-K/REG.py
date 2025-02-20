"""
setup_rdp.py v0.1
F. Falcioni, P. L. A. Popelier

Script with the function to run the Ramer-Douglas-Peucker algorithm to potential energy surfaces.

Check for updates at github.com/FabioFalcioni

Please, report bugs and issues to fabio.falcioni@manchester.ac.uk
coded by F.Falcioni
"""
from pickle import TRUE
import reg
import numpy as np
plt.switch_backend('agg')
import matplotlib.pyplot as plt
from optparse import OptionParser
import csv
import os
import re
import pandas as pd

def get_xyz_coords(file_loc):
    scan_pattern = r'(\s*\!\s+[A-Za-z]+\s+[A-Z]+[(\[][0-9\,]+[)\]])\s+([0-9.]+)\s+Scan\s+!'
    cc = []
    converged_step = False
    coord_pattern = r'\s+\d+\s+(\d+)\s+\d\s+(-?\d+.\d+)\s+(-?\d+.\d+)\s+(-?\d+.\d+)\s+'
    xyz_coordinates = []
    list_of_xyz_coordinates = []
    stop_index = 0
    opt=False
    energy_pattern = r' SCF Done:  E[(\[]([A-Za-z \d]+)[)\]]\s*=\s*(-?\d+.\d+)'
    energy = []
    theory_level = []

    with open(file_loc,'r') as file:
        lines = file.readlines()
        for i,line in enumerate(lines):
            matched = re.search(scan_pattern,line)
            cc_opt = re.search(r'GICDEF:\s*([A-Za-z][(\[][0-9\,]+[)\]]+)',line)
            if matched:
                scan_match = r'{}\s+([0-9.]+)\s*-DE/DX =\s*'.format(matched.group(1).replace('(','\(').replace(')','\)'))
                stop_index = i
                break
            elif cc_opt:
                scan_match = r'{}\s+([0-9.]+)'.format(cc_opt.group(1).replace('(','\(').replace(')','\)'))
                stop_index = i
                opt = True
                break  

        for line in reversed(lines[stop_index+10:]):
            cc_matched = re.search(scan_match,line)
            energy_matched = re.search(energy_pattern,line)
            if cc_matched:
                cc.append(cc_matched.group(1))
                converged_step = True
            elif converged_step and re.match(coord_pattern,line):
                xyz_coordinates.append(re.sub(coord_pattern, r'\1   \2  \3  \4 \n',line))
            elif converged_step and energy_matched:
                theory_level.append(energy_matched.group(1))
                energy.append(energy_matched.group(2))
            elif converged_step and re.search(r' Number     Number       Type             X           Y           Z',line):
                list_of_xyz_coordinates.append(xyz_coordinates)
                xyz_coordinates = []
                converged_step = False
                if opt:
                    break

    return cc,energy,theory_level,list_of_xyz_coordinates

'''
Method for looking for and loading all .out or .log files in all sub folders. 'Hey im walking here'
'''
def find_outputs():
    out_files = []
    for root,_,files in os.walk("."):
        for name in files:           
            if name.endswith(".out") or name.endswith(".log"):
                out_files.append(os.path.join(root,name))

    return out_files

'''
Method for creating dataframe of data from output files
'''
def data_compile(cc_list,energy_list,theory_level_list,xyz_list):
    data = {}
    ccs = []
    energies = []
    theory_levels = []
    xyzs = []

    for i,cc in enumerate(cc_list):
        ccs.append(cc)
        energies.append(energy_list[i])
        theory_levels.append(theory_level_list[i])
        xyzs.append(xyz_list[i])

    data['Control coord'] = ccs
    data['energy'] = np.array(energies,dtype=float)
    data['Level of theory'] = theory_levels
    data['XYZ_Coords'] = xyzs

    data_df = pd.DataFrame(data).drop_duplicates(subset='Control coord').sort_values('Control coord')
    data_df['energy'] = data_df["energy"].transform(lambda x: (x - min(data['energy'])) * 2625.5) #Converting to kJmol

    return data_df

'''
Method to write an XYZ file for each control coordinate.
'''
def write_XYZ_methods(data_frame,out_dir):
    #Creating directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for index, row in data_frame.iterrows():
        output = ''
        output += str(len(row['XYZ_Coords'])) + '\n'
        output += 'Step: {}  E({}): {} \n'.format(row['Control coord'], row['Level of theory'], row['energy'])
        for coord in reversed(row['XYZ_Coords']):
            output += coord
        new_file = open('{}/step_{}.xyz'.format(out_dir,str(row['Control coord'])),'w')
        new_file.write(output)
        new_file.close()


def scan_files_and_combile_data():
    full_cc = []
    full_energy = []
    full_theory_level = []
    full_coords = []

    files = find_outputs()

    for file in files:
        cc, energy, theory_level, list_of_xyz_coordinates = get_xyz_coords(file)
        full_cc = full_cc + cc
        full_energy = full_energy + energy
        full_theory_level = full_theory_level + theory_level
        full_coords = full_coords + list_of_xyz_coordinates

    data_frame = data_compile(full_cc,full_energy, full_theory_level, full_coords)
    return data_frame


def re_fit_rdp(X,Y):
    X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
    max_Y = max(Y)
    min_Y = min(Y)
    delta_Y_av = (max_Y - min_Y)/(len(Y)-1)
    delta_X_av = (max(X)-min(X)) / (len(X)-1)
    delta_X_scale = delta_Y_av / delta_X_av
    
    new_X = {min_Y : X[0]}

    for i in range(len(X)-1):
        delta_X = X[i+1] - X[i]
        new_delta_X = delta_X * delta_X_scale
        new_X_val = list(new_X.keys())[i] + new_delta_X
        new_X[new_X_val] = X[i+1]

    return list(new_X.keys()),Y,new_X


def find_point_between_two_points(start, end, value):
    '''
    Function to find the y value of a point between two other points of a function given its x value (i.e. interpolation)
    '''
    slope = slope_intercept(start[0], start[1], end[0], end[1])
    intercept = start[1] - slope * start[0]
    y = slope * value + intercept
    return slope


def slope_intercept(x1, y1, x2, y2):
    '''
    Function to find the slope and intercept of a line between two points
    '''
    a = (y2 - y1) / (x2 - x1)
    return a


def deviation(point, start, end):
    '''
    Function to calculate the perpendicular distance of a point from the line defined by other two points.
    '''
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start))


def rdp_rec(M, epsilon, pdist=deviation):
    '''
    Function to run the RDP algorithm recursively
    '''
    max_dist = 0.0
    index = -1

    for i in range(1, M.shape[0]):
        d = pdist(M[i], M[0], M[-1])

        if d > max_dist:
            index = i
            max_dist = d
    if max_dist > epsilon:
        r1 = rdp_rec(M[:index + 1], epsilon, pdist)
        r2 = rdp_rec(M[index:], epsilon, pdist)
        return np.vstack((r1[:-1], r2))
    else:
        return np.vstack((M[0], M[-1]))


def rdp(M, epsilon=0, pdist=deviation):
    if "numpy" in str(type(M)):
        return rdp_rec(M, epsilon, pdist)
    return rdp_rec(np.array(M), epsilon, pdist).tolist()


def find_maximum_deviation(M, pdist=deviation):
    '''
    Function to calculate the maximum perpendicular distance from the line defined between the extreme points of a segment
    '''
    max_dist = 0.0
    index = -1

    for i in range(1, M.shape[0]):
        d = pdist(M[i], M[0], M[-1])

        if d > max_dist:
            index = i
            max_dist = d
    return max_dist


def rmse(true_value, predicted_value):
    '''
    Function to calculate the Root-Mean-Squared-Error between true and predicted values
    '''
    if len(true_value) != len(predicted_value):  ## Checks if X and Y have the same size
        raise ValueError("Arrays must have the same size")
    error = [true_value[i] - predicted_value[i] for i in range(len(true_value))]
    temp = [a ** 2 for a in error]
    RMSE = (sum(temp) / len(temp)) ** 0.5
    return RMSE


def minimum_points_segment_RDP(energy, cc, epsilon):
    '''
    Function to obtain a segment (or function) with the minimum amount of points given a specific value of epsilon (i.e. deviation)
    '''
    vector = []
    x = []
    y = []

    for i in range(0, len(energy)):
        coordinate = (cc[i], energy[i])
        vector.append(coordinate)
    new_points = rdp(vector, epsilon=epsilon)
    for value in new_points:
        new_x, new_y = value
        x.append(new_x)
        y.append(new_y)
    new_interpolated_points = []
    for j in range(1, len(new_points)):
        for i in range(0, len(cc)):
            if i > cc.index(new_points[j - 1][0]) and i < cc.index(new_points[j][0]):
                interpolate_point = find_point_between_two_points(new_points[j - 1], new_points[j], cc[i])
                new_interpolated_points.append(interpolate_point)
    y_reference = y + new_interpolated_points
    if energy[0] > energy[-1]:
        y_reference.sort(reverse=True)
    else:
        y_reference.sort(reverse=False)
    RMSE = rmse(energy, y_reference)
    return y, x, RMSE


def cross_validation_RDP(energy, cc, epsilon = 0.5, step_size=0.01):
    '''
    Function to run the RDP algorithm X times (depending on step_size) at different values of epsilon and obtain
    obtain a function with lowest number of points at that (or below) RMSE of confidence.
    '''
    new_vector = []
    #cc = [cor* for cor in cc]
    [new_vector.append((cc[i], energy[i])) for i in range(0, len(energy))]
    max_epsilon = find_maximum_deviation(np.array(new_vector), pdist=deviation)
    y_values_minimum = []
    x_values_minimum = []
    rmse_min = []
    y_values, x_values, rmse = minimum_points_segment_RDP(energy, cc, epsilon)
    if True:
        y_values_minimum.append(y_values)
        x_values_minimum.append(x_values)
        rmse_min.append(rmse)

    return min(x_values_minimum, key=len), min(y_values_minimum, key=len), rmse_min[
        x_values_minimum.index(min(x_values_minimum, key=len))]


def main():
    #Parsing USER INPUT
    global segments, cc_new
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-r", "--epsilon", action='store', type='float', dest='epsilon_val',
                      help="Root Mean Squared error of confidence")
    
    try:
        (option, args) = parser.parse_args()
        eps_val = option.repsilon_val
    except:
        eps_val = 0.3

    print(
        "RDP setup: searching for a new polyline with epsilon of {} ...".format(eps_val))
    
    scan_df = scan_files_and_combile_data()


    wfn_energies = np.array(scan_df['energy'],dtype=float)
    cc = np.array(scan_df['Control coord'],dtype=float)

    cc,wfn_energies, x_dict = re_fit_rdp(cc, wfn_energies) # Scaling the coordinates so the RDP works well

    og_cc = list(x_dict.values())

    # Search for critical points in the function
    if TRUE:
        tp = reg.find_critical(wfn_energies, cc, use_inflex=False, min_points=3)
        segments = reg.split_segm(wfn_energies, tp)
        cc_new = reg.split_segm(cc, tp)
        '''
    elif isinstance(option.critical_points, str):
        tp = [(int(value)) for value in option.critical_points.split(',')]
        segments = reg.split_segm(wfn_energies - sum(wfn_energies) / len(wfn_energies), tp)
        cc_new = reg.split_segm(cc, tp)
    elif not option.critical_points:
        segments = [wfn_energies - sum(wfn_energies) / len(wfn_energies)]
        cc_new = [cc]'''

    all_selected_points = []
    #Creating a figure with the new polyline given the USER input rmse of confidence
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=( 9,5))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    for i,segment in enumerate(segments):
        if len(segment) > 2:
            x, y, RMSE = cross_validation_RDP(segment, cc_new[i], epsilon = eps_val)
            x = [x_dict[i] for i in x]
        else:
            x = cc_new[i]
            x = [x_dict[i] for i in x]
            y = segment
            RMSE = 0.0
        print("RMSE Segment {} = {}".format(i + 1, RMSE))
        print("Points of the new polyline for Segment {} = {}".format(i + 1, x))
        plt.plot(x, y, marker='o', markersize=10)
        plt.plot(x[0], y[0], marker='o', markersize=10, c='red')
        plt.plot(x[-1], y[-1], marker='o', markersize=10, c='red')

        all_selected_points += y


    plt.plot(og_cc, wfn_energies, c='#4d4d4d', marker='o', markersize=2)
    plt.xlabel(r'Control Coordinate (Å)')
    plt.ylabel(r'Relative Energy (kJ $\mathrm{mol^{-1}}$)')
    fig.savefig('RDP_out.png', dpi=300, bbox_inches='tight')

    # Removing RDP geomerties for saving
    select_rdp_df = scan_df[scan_df['energy'].isin(all_selected_points)]
    removed_rdp_df = scan_df[~scan_df['energy'].isin(all_selected_points)]
    scan_df['RDP_select'] = scan_df['energy'].isin(all_selected_points)
    scan_df[['Control coord', 'energy','RDP_select']].to_csv('PES_Scan.csv')
    write_XYZ_methods(select_rdp_df,'Selected_Points')
    write_XYZ_methods(removed_rdp_df,'Removed_Points')

if __name__ == "__main__":
    main()

import os
import sys

def Find_Completed():
    '''
    Finds all int files in sub directories and adds them to a dictionary.
    
    The key of the dictionary represents the step in the PES and the value is the list of int files calculated for that step.
    '''
    interaction_files = []
    wfn_file = []
    root_dirs = []
    for root,_,files in os.walk("."):
        for name in files:
            if name.endswith(".int"):
                file_loc = os.path.join(root,name)
                with open(file_loc,'r') as f:
                    if "AIMInt is Done." in f.read():
                        interaction_files.append(file_loc[:-4] + '.inp')
            
            elif name.endswith(".wfn"):
                wfn_file.append(os.path.join(root,name))
                root_dirs.append(os.path.join(root,"frame_atomicfiles"))

    return interaction_files, wfn_file, root_dirs

def AddRoot(file_list, existing_int_atoms,roots):
    '''
    Adds the root directory and removes calculations that have already been run.
    '''
    run_file_dirs = []
    for dirs in roots:
        for f in file_list:
            run_file_dirs.append(os.path.join(dirs,f))
    run_file_dirs = [file for file in run_file_dirs if file not in existing_int_atoms]
    return run_file_dirs

def get_atom_list(wfn_file):
    """
    ###########################################################################################################
    FUNCTION: get_atom_list
              get atomic labels from wfn file

    INPUT: wfn_file
        wfn_file = Any wfn file of the desired PES

    OUTPUT: atom_list
        list of each atom label for all atoms in molecule

    ERROR:
        "Atomic labels not found" : Atom list does not exist in wfn_file
    ###########################################################################################################
    """

    # INTERNAL VARIABLES:
    atom_list = []

    # OPEN FILE:
    file = open(wfn_file, "r")
    lines = file.readlines()  # Convert file into a array of lines
    file.close()  # Close file

    # ERRORS:
    if "(CENTRE " not in lines[2]:
        raise ValueError("Atomic labels not found")  # Checks if atomic list exist inside file

    # GET ATOM LIST:
    for i in range(len(lines)):
        if "(CENTRE" in lines[i]:
            split_line = lines[i].split()
            atom_list.append(split_line[0].lower() + str(split_line[1]))  # uppercase to lowercase

    return atom_list


def Interactions(atom_list, existing_int_atoms, wfn_root, root_dirs):
    '''
    Find out which interactions will need to be run.

    Return list of intra .inp files to run.
    Return list of inter .inp files to run.
    '''
    intra_files = []
    inter_files = []
    atoms = get_atom_list(wfn_root)

    if len(atom_list) > 0:
        new_atoms = []
        for atom in sorted(atom_list):
            new_atoms.append(atoms[atom - 1])
        atoms = new_atoms

    for atom1 in atoms:
        for atom2 in atoms:
            if atom1 == atom2:
                intra_files.append(str(atom1) + ".inp")
            elif int(atom1[1:]) < int(atom2[1:]):
                inter_files.append(str(atom1) + '_' + str(atom2) + '.inp')
    
    inter_files = list(dict.fromkeys(inter_files))

    root_inter_files = AddRoot(inter_files,existing_int_atoms,root_dirs)
    root_intra_files = AddRoot(intra_files,existing_int_atoms,root_dirs)

    return root_intra_files, root_inter_files
        
def CreateAimint(files, wfn_files, n_p_task, aimall_loc, max_cores):
    '''
    Generate Job files to run for all interactions in aimaill as .sh submission

    files = intra + inter files combined
    wfn_files = locations of wfn files to use in aimall
    n_p_task = number of cores per task
    aimall_loc = location of aimall program on system
    max_cores = maximum number of cores you want the array to take up
    '''
    tc = int(max_cores) // int(n_p_task)
    no_total_tasks = len(files)

    if tc > no_total_tasks:
        tc = no_total_tasks

    out_put = "#!/bin/bash --login \n#$ -S /bin/bash \n#$ -cwd \n#$ -pe smp.pe {} \n#$ -t 1-{} \n#$ -tc {} \n\n".format(str(n_p_task), str(no_total_tasks),str(tc))
    out_put += "JOB_ARRAY=(\n"

    for wfn in wfn_files:
        for file in files:
            if wfn[:4] == file[:4]:
                out_put += '"' + " -nproc={} ".format(n_p_task) + str(file) + " " + str(wfn) + '"\n'
    
    out_put += ")\n\n"
    out_put += "TID=$[SGE_TASK_ID-1]\n\n"
    out_put += "IDX=$[TID%{}]\n\n".format(no_total_tasks)
    out_put += "JOBID=${JOB_ARRAY[$IDX]}\n\n"
    out_put += str(aimall_loc) + " $JOBID"
    
    return out_put

def CreateAimQB(wfn_files,n_p_task,aimall_loc,max_cores,atoms=[]):
    '''
    Writing mesh file if you want to generate mesh in Aimall in parallel
    
    wfn_files = locations of wfn files to use in aimall
    n_p_task = number of cores per task
    aimall_loc = location of aimall program on system
    max_cores = maximum number of cores you want the array to take up
    '''
    tc = int(max_cores) // int(n_p_task)
    if len(atoms) > 0:
        no_total_tasks = len(wfn_files) * len(atoms)
    else:
        no_total_tasks = len(wfn_files)

    if tc > no_total_tasks:
        tc = no_total_tasks

    out_put = "#!/bin/bash --login \n#$ -S /bin/bash \n#$ -cwd \n#$ -pe smp.pe {} \n#$ -t 1-{} \n#$ -tc {} \n\n".format(str(n_p_task), str(no_total_tasks),str(tc))
    out_put += "JOB_ARRAY=(\n"

    if len(atoms) == 0:
        for wfn in wfn_files:
            out_put += '"' + " -nproc={} ".format(n_p_task) + " " + str(wfn) + '"\n'
    else:
        for wfn in wfn_files:
            for atom in atoms:
                out_put += '"' + " -atoms={} -nproc={} ".format(atom,n_p_task) + " " + str(wfn) + '"\n'
        
    out_put += ")\n\n"
    out_put += "TID=$[SGE_TASK_ID-1]\n\n"
    out_put += "IDX=$[TID%{}]\n\n".format(no_total_tasks)
    out_put += "JOBID=${JOB_ARRAY[$IDX]}\n\n"
    out_put += str(aimall_loc) + " -nogui -iamesh=superfine -delmog=false -encomp=4 -skipint=true $JOBID"
    return out_put
     
if __name__=="__main__":
    '''
    Variables.
    '''
    atom_selection = []
    aim_path = '/mnt/iusers01/pp01/w06498gk/AIMAll/aimint.ish' # Path for aimint
    aimqb_path = '/mnt/iusers01/pp01/w06498gk/AIMAll/aimqb.ish' # Path for aimqb
    np_per_task_inter =str(2) # Number of cores for inter atomic calcs
    np_per_task_intra =str(8) # Number of cores for intra atomic calcs
    max_cores = str(200) # Miximum number of cores you can use
    redo_aims = [] # Which aimall integrations do you want to redo?
    drop_intra = False # Removing intra calcs from aim_inter file
    AIMINTRA = False # Write aim intra file
    AIMINTER = True # Write aim inter file
    NoOverWrite = False # Do not overwright written aim input files
    MESH = False # Write mesh files True/False
    
    completed_aims, wfn_files, root_dirs = Find_Completed()
  
    if redo_aims == all:
        completed_aims = []
    elif len(redo_aims) > 0:
        for redo in redo_aims:
            for aim in completed_aims:
                if str(redo) in aim:
                    completed_aims.remove(aim)

    intra_files, inter_files = Interactions(atom_selection, completed_aims, wfn_files[0], root_dirs)
   
    if AIMINTRA:
        output_intra = CreateAimint(intra_files, wfn_files, np_per_task_intra, aim_path, max_cores)
        f = open("aim_intra.sh", "w")
        f.write(output_intra)
        f.close()

    if AIMINTER:
        if drop_intra:
            for intra_file in intra_files:
                split_intra = intra_file.split('/')
                for inter_file in inter_files:
                    containing_folder = '/' + str(split_intra[-3]) + '/'
                    if split_intra[-1].split('.')[0] in inter_file and containing_folder in inter_file:
                        inter_files.remove(inter_file)
        output_inter = CreateAimint(inter_files, wfn_files, np_per_task_inter, aim_path, max_cores)
        if NoOverWrite:
            f = open("aim_inter1.sh", "w")
        f = open("aim_inter.sh", "w")
        f.write(output_inter)
        f.close()

    if MESH:
        output_mesh = CreateAimQB(wfn_files,str(8),aimqb_path, max_cores)
        f = open("aim_mesh.sh", "w")
        f.write(output_mesh)
        f.close()



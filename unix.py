#!/usr/local/bin/python2.7

import subprocess as subp
import configobj
import copy
import os
import sys
import time

global GPUID
GPUID=0

def create_folders_for_parameters(input_file, output_file):
    
    start_time = time.time()
    # import config file:
    cfg = configobj.ConfigObj(input_file, write_empty_values=True, list_values=True, interpolation=True,stringify="False")
    p = copy.deepcopy(cfg)

    # select those parameters, which have more than one value:
    var_par={}
    # save the section name of the variable in "sec_var" dictionary:
    sec_var={}

    for sec in p:
        for var in p[sec]:

            tmp=list(p[sec][var])

            if len(tmp)>1:
                var_par[var]=tmp
                sec_var[var]=sec
            else:
                p[sec][var]=tmp[0]

    # num_of_folders: the total number of subfolders have to create, or the product of len() changing variable values:
    num_of_folders=1
    # dict_len holds the length of the individual changing variable lists:
    dict_len={}
    # number of variables that have more values:
    var_num=len(var_par.keys())

    if var_num>0:
        print("{} variables to replace!".format(var_num))

        for key in var_par.keys():

            tmp=len(var_par[key])
            dict_len[key]=tmp
            num_of_folders=num_of_folders*tmp

     # this loop creates the foldernames, identified each by the variable name and by the value, "."-s are replaced by E:

        for i in range(num_of_folders):
            foldername=""
            denumerator=1
            tmp_obj=copy.deepcopy(p)
            for var in range(var_num):
                key=var_par.keys()[var]

                index=(i/denumerator)%dict_len[key]
                value=list(var_par[key])[index]
                foldername=foldername+"_{}_{}".format(key,value)

                denumerator=denumerator*dict_len[key]
      # replace the list of values in the config file with the single value choosen for the simulation:
                sect_name=sec_var[key]
                tmp_obj[sect_name][key]=value
      # replace the "."-s with E in the foldername:
            foldername=foldername.replace(".","E")
            foldername=foldername.replace("/","div")
            foldername=foldername.replace("*","x")
            print foldername

      # create the folder called foldername:
            subp.call(["mkdir",foldername])
      # save the new config file:
            filename=output_file
            tmp_obj.filename=filename
            tmp_obj.write()
      # copy the necessary files to the proper folder:
            subp.call(["mv",filename,foldername])
            subp.call(["cp","maggrad_ocl",foldername])
            subp.call(["cp","kernel.cl",foldername])
            subp.call(["cp","init_functions.cl",foldername])

      # run the simulation in the subfolder:
            wd=os.getcwd()
            os.chdir(foldername)
            cmd="./maggrad_ocl --gpu "
            cmd=cmd+str(GPUID)
            os.system(cmd)
            os.chdir(wd)
            print "------------------------------"
            print "\t END OF THE SIMULATION"
            print "Time consumed by the simulation: {}".format(time.time()-start_time)
            print "------------------------------"

      # if there were no list values in the config file, we don't need to create subfolders, simulation is started in the current directory:
    else:
        print("No variable with multiple values!")
        filename=output_file
        p.filename=filename
        p.write()
        cmd="./maggrad_ocl --gpu "
        cmd=cmd+str(GPUID)
        print cmd
#        proc=subp.Popen(["maggrad_ocl"])
        os.system(cmd)


def main():
    create_folders_for_parameters('parameters_binary_tmp.cfg','parameters_binary.cfg')


if __name__ == '__main__':
    try:
      main()
    except KeyboardInterrupt:
      pass

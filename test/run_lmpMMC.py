import os
import numpy as np
import shutil

from MMC.mmc import DFWriter, LOGWriter, PyLMP4MMC, MMC


fsummary = "mySummary.csv"
flogfile = "myMMC.log"
SummaryDF = DFWriter(fsummary)
Logfile = LOGWriter(flogfile)

DataOut_Path = "DataOut"
if not os.path.isdir(DataOut_Path):
    os.makedirs(DataOut_Path)


EREFs = None 
ff_elements = np.array(["Fe", "Fe", "Cr"])
Exclude_types = None #[3, 4]
Enforce_types = [2]
ntypes = len(ff_elements)

infile = "in.MMC"
fdata = "2lp_100_5Cr.data"
ratio_hot = 0.3
ratio_cold = 0.4
reverse_cold = True
mydata = MMC(ntypes, EREFs=EREFs, ff_elements=ff_elements,
             ratio_hot=ratio_hot, ratio_cold=ratio_cold, reverse_cold=reverse_cold)
shutil.copy(fdata, os.path.join(DataOut_Path, "MMC0.dat"))

tol = 0.001
loopmax = 10000
nsteps4relax = 100
nsteps4writedata = 100
nsteps4checkpoint = 100
nsteps4visual = 10
nsteps4summary = 10
nsteps4updateEREFs = 100
Temperature = 300.0

Screen = False #True
Log = False #"log.lammps"
lmp = PyLMP4MMC(Screen=Screen, Log=Log)

iloop = 0
iaccept = 0
ireject = 0
lmp.excute_file(infile)
mydata.last_TE, mydata.last_types = lmp.get_total_energy_types(iloop)
mydata.natoms = len(mydata.last_types)
eatoms = lmp.get_eatoms(iloop, mydata.natoms)
mydata.update_EREFs(mydata.last_types, eatoms)

ratio_shift = ratio_hot
atom_style = "atomic"
mydata.write_shifted_data(mydata.last_types, eatoms, ratio_shift, "MMC" + str(iloop) + ".dat", atom_style=atom_style)

init_energy = mydata.last_TE
energy_checkpoint = init_energy


logstr = f"loopmax:{loopmax} Temp:{Temperature} convergenc: energy < {tol} in {nsteps4checkpoint} steps"
print(logstr)
Logfile.write_to_file(logstr, open_style="w")
logstr = f"start MMC for fname:{fdata} natoms:{mydata.natoms} ratio_hot:{ratio_hot} ratio_cold: {ratio_cold}"
print(logstr)
Logfile.write_to_file(logstr, open_style="a")
logstr = f"Reference energies of each type at {iloop} step are :{mydata.EREFs}"
print(logstr)
Logfile.write_to_file(logstr, open_style="a")
logstr = f"== iloop:{iloop} iaccept:{iaccept} ireject: {ireject} total_energy:{mydata.last_TE} =="
SummaryDF.append_to_file(iloop, iaccept, ireject, init_energy)
print(logstr)
Logfile.write_to_file(logstr, open_style="a")

isValid = True
while isValid:
    id_hot, id_cold = mydata.get_select_ids(mydata.last_types, eatoms, Exclude_types=Exclude_types, Enforce_types=Enforce_types, maxloop=1000)
    this_types = mydata.get_this_types(id_hot, id_cold)
    lmp.scatter_this_types(this_types)
    iloop  += 1
    mydata.this_TE, mydata.this_types = lmp.get_total_energy_types(iloop)
    eatoms = lmp.get_eatoms(iloop, mydata.natoms)

    if iloop % nsteps4updateEREFs == 0:
        mydata.update_EREFs(mydata.this_types, eatoms)
        logstr = f"Reference energies of each type at {iloop} step are :{mydata.EREFs}"
        print(logstr)

    isAccept, iaccept, ireject = mydata.MMC(iaccept, ireject, Temp=Temperature)
    if not isAccept:
        lmp.scatter_this_types(mydata.last_types)

    if iloop % nsteps4writedata == 0:
        lmp.write_data(iloop)

    if iloop % nsteps4summary == 0:
        SummaryDF.append_to_file(iloop, iaccept, ireject, mydata.last_TE)

    if iloop % nsteps4visual == 0:
        logstr = f"== iloop:{iloop} iaccept:{iaccept} ireject: {ireject} total_energy:{mydata.last_TE} =="
        print(logstr)
        Logfile.write_to_file(logstr, open_style="a")
    
    isValid, energy_checkpoint = mydata.checkpoint(iloop, isValid, energy_checkpoint,
                                                   nsteps4checkpoint=nsteps4checkpoint, tol=tol,
                                                   loopmax=loopmax)

lmp.write_data(iloop)
lmp.close()
logstr = f"== iloop:{iloop} iaccept:{iaccept} ireject: {ireject} total_energy:{mydata.last_TE} =="
print(logstr)
Logfile.write_to_file(logstr, open_style="a")
SummaryDF.append_to_file(iloop, iaccept, ireject, mydata.last_TE)


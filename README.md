# GDPM: an Algorithm for Gradual Discovery in Pattern Mining    
## PKDD-2020 experiments
---
Read this instruction to replicate experiments. To select optimal patterns among the generates ones we use [Krimp](https://people.mmci.uni-saarland.de/~jilles/prj/krimp/). The experiments includes intermediate procedures to (re)convert data to a Krimp-readable format.

## Pipeline

1. Data preparation
    - Download data and remove class labels
    - Convert datasets without labels to *.db*-file (use Krimp).
2. Computing n-closed patterns on unlabeled datasets 
    - Load a *.db*-dataset as [numpy.ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)  (to convert it use `modules/io/dbconverters/db2bindata`)
    - Run `modules/binary/n_closed/compute_closed_by_level` and check the results in `./results/` folder. Computed patterns are stored in [pickle](https://docs.python.org/3/library/pickle.html) binary files. 
    - Convert patterns to *.isc* format (`modules/io/dbconverters/write_isc`)
5. Run Krimp with the computed candidates stored in *.isc* file.
6. Analyse the results

The code corresponding to steps 2 and 6 is available [here](https://github.com/tmghub/pattern_mining_tools/blob/master/experiments/PKDD_2020/computing_and_summarising.py).

## Data source
The datasets can be download from [LUCS-KDD](https://cgi.csc.liv.ac.uk/~frans/KDD/Software/LUCS-KDD-DN/DataSets/dataSets.html). We use the following ones:

| Name | #objects | #attributes|
| ------ | ------ |------ |
|auto.D137.N205.C7.num|205|129|
|breast.D20.N699.C2.num|699|14|
|car.D25.N1728.C4.num|1728|21|
|ecoli.D34.N336.C8.num|326|24|
|glass.D48.N214.C7.num|214|40|
|heart.D52.N303.C5.num|303|45|
|hepatitis.D56.N155.C2.num|155|50|
|iris.D19.N150.C3.num|150|32|
|led7.D24.N3200.C10.num|3200|28|
|penDigits.D89.N10992.C10.num|768|36|
|soybean-large.D118.N683.C19.num|683|99|
|ticTacToe.D29.N958.C2.num|958|27|
|wine.D68.N178.C3.num|178|65|
|zoo.D42.N101.C7.num|101|35|

The class labels were removed from datasets. The datasets should be stored in [numpy.ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html). 

## Computing optimal patterns with Krimp
---
Optimal patterns are computed by [Krimp](https://people.mmci.uni-saarland.de/~jilles/prj/krimp/). Candidates to optimal patterns should be written in '.ics'-files. To convert pattern to this format use `./modules/io/write_isc` function. Then, to run Krimp-compression by this pattern-candidates  use the following parameters in `compress.conf` file (for other parameters see Krimp documentation):

```
taskclass = main
command = compress
takeItEasy = 0

dataDir = <input folder>
expDir = <output folder>
dbName = <dataset name>
iscName = <patterns converted by `./modules/io/write_isc`>

iscIfMined = store
pruneStrategy = pop
dataType = bai32
numThreads = 1 
algo = coverpartial
reportSup = 50
reportCnd = 0
reportAcc = true 
iscStoreType = isc
iscChunkType = isc
```


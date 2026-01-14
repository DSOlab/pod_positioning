# pod_positioning

Tiny command-line tool to run POD from DORIS RINEX using SINEX (DPOD), VMF3
and SP3 inputs.

Build: 

Assuming that necessary libraries (integrator, rwatmo, sysnsats, sinex, sp3, iers, cspice) are installed in the system, the following command should build the executable for pod.

```
g++ -Wall -std=c++17 pod.cpp -lintegrator -lyaml-cpp -lrwatmo -lrnx -lsysnsats -lsinex -lsp3 -liers -lcspice -lgeodesy -ldatetime -o pod
```

Requirements: Eigen3, yaml-cpp

Usage:

```bash
pod <config.yaml> <rnx.obs> <dpod.snx> <dpod_freq.txt> <vmf3_dir> <sp3c>
```
 
## Note

Large auxiliary files (VMF3 grids, DPOD/SINEX, EOP/AOD1B, SP3 archives) are
not stored in this repo. 
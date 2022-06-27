#!/bin/sh

fName='./trajecStatus.txt'
rm -f _rslurm_BEA*/core.[0-9][0-9]* >/dev/null 2>&1
rm -f out/by-id/*/core.[0-9][0-9]* >/dev/null 2>&1
rm -f ${fName} >/dev/null 2>&1
date > ${fName}
echo "----------------------------" >> ${fName}
echo "Errors:       `ls out/by-id/*/ERROR | wc -l`" >> ${fName}
echo "Obs files:    `ls out/obs/ | wc -l`" >> ${fName}
echo "Footprints:   `ls out/footprints/ | wc -l`" >> ${fName}
echo "Trajectories: `ls out/particles/ | wc -l`" >> ${fName}
echo "Directories:  `ls out/by-id/ | wc -l`" >> ${fName}
cat ./trajecStatus.txt

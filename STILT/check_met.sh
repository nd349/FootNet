#!/bin/sh
cd ~/STILT/BEACON/.
fName='./metStatus.txt'
date > ${fName}
echo "----------------------------" >> ${fName}
hh=0
for hh in 0 6 12 18; do
   echo " " >> ${fName}
   echo "  ${hh}z" >> ${fName}
   echo "---------------" >> ${fName}
   yyyy=2018
   while [ $yyyy -le 2020 ]; do
      mm=1
      #echo "*** $yyyy   : $(ls $(printf './met_data/hrrr_trim/hysplit.%04d*.%02dz.hrrra' $yyyy $hh) | wc -l)" >> ${fName}
      echo "*** $yyyy   : $(ls $(printf '/clusterfs/aiolos/aturner/met_BayArea/hrrr_trim/hysplit.%04d*.%02dz.hrrra' $yyyy $hh) | wc -l)" >> ${fName}
      while [ $mm -le 12 ]; do
         mmS=$mm
         if [ $mm -lt 10 ]; then mmS="0${mm}"; fi
         #echo "  * $mmS/$yyyy: $(ls $(printf './met_data/hrrr_trim/hysplit.%04d%02d*.%02dz.hrrra' $yyyy $mm $hh) | wc -l)" >> ${fName}
         echo "  * $mmS/$yyyy: $(ls $(printf '/clusterfs/aiolos/aturner/met_BayArea/hrrr_trim/hysplit.%04d%02d*.%02dz.hrrra' $yyyy $mm $hh) | wc -l)" >> ${fName}
         mm=$(( $mm + 1 ))
      done
      yyyy=$(( $yyyy + 1 ))
      echo " " >> ${fName}
   done
done
cat ${fName}

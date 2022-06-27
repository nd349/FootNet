### ====================================================
### = read_Background.r
### = Alex Turner
### = 03/06/2020
### =---------------------------------------------------
### = ( 1) traj     :: Array containing the trajectory.
### =    - time     : Elapsed time in hours (negative).
### =    - index    : Particle index.
### =    - lat      : Latitude of the particle.
### =    - lon      : Longitude of the particle.
### =    - agl      : Altitude [m]
### =    - foot     : Footprint
### =    - zi       : Mixed layer height [m]
### =    - dmass    : Mass violation
### = ( 2) run_time :: POSIXt date for the receptor.
### = ( 3) debug    :: Print the diagnostics?
### =---------------------------------------------------
### = NOTES:
### = ( 1) Read the NOAA curtain and figure out the 
### =      background value for the particle.
### ====================================================

### Function
find_background <- function(traj=NULL,run_time=NULL,debug=FALSE)
{

### Define some directories and parameters
noaaDir = '/global/home/users/aturner/STILT/BEACON/NOAA_Curtain'
nasaDir = '/global/home/users/aturner/STILT/BEACON/NASA_CO2'
co2name = 'co2.pacific.bgarr.Rda'

### Specify the Kriging length scales
lx <- 50	# km in horiz
ly <- 1		# km in vert

### Arylns functions
decimal.to.POSIX <- function(dd,tz="UTC") {
  # converts a vector of YYYY.yyyy decimal dates
  # such as those used by Arlyn into POSIXt dates.
   year <- trunc(dd)
    y0 <- ISOdatetime(year,1,1,0,0,0,tz="UTC")
    y1 <- as.numeric(ISOdatetime(year+1,1,1,0,0,0,tz="UTC"))
    sec.in.year <- y1 - as.numeric(y0)
    pd <- as.POSIXct(y0 + (dd-year)*sec.in.year,tz=tz)
   return(pd)
}
POSIX.to.idate <- function(dates,tz="UTC") {
  lt <- as.POSIXlt(dates,tz=tz)
  return(data.frame(year=lt$year+1900,
                    month=lt$mon+1,
                    day=lt$mday,
                    hour=lt$hour,
                    minute=lt$min,
                    second=lt$sec))
}

### Get info about the trajectory
# Get the time into a julian date
prinForm  <- "%04i-%02i-%02i %02i:00:00"
origin    <- "1970-01-01"
# Save the important data from the trajectory
lenP      <- dim(traj)[1]
#part.time <- traj[,"time"]/(24*60) + as.numeric(base::julian(run_time,origin=origin))
#part.time <- base::julian(traj[,"time"]*60 + run_time,origin=origin)
part.time <- traj[,"time"]*60 + run_time
part.lat  <- traj[,"lati"]	# deg
part.lon  <- traj[,"long"]	# deg
part.agl  <- traj[,"zagl"]/1000 # km
part.ind  <- traj[,"indx"]	# particle index

### ========================
### = Load the NOAA Curtain
### ========================

### Load the curtain
load(paste(noaaDir,'/ROBJ/',co2name,sep=""))
time = ddbg			# decimal year
lat  = latbg			# deg
alt  = altbg * 1e-3		# km
co2  = co2.pacific.bgarr	# ppm
err  = co2.pacific.bgrms	# ppm
# Convert the NOAA time to a julian date
#time <- base::julian(decimal.to.POSIX(time),origin=origin)
time <- decimal.to.POSIX(time)
# Save the important NOAA data
noaa.time = time	# julian date
noaa.lat  = lat		# deg
noaa.alt  = alt		# km
noaa.co2  = co2		# ppm
noaa.err  = err		# ppm

### ========================
### = Get the NASA background
### ========================

### NASA pressure levels
Ap <- c(0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
        3.257081e+01, 3.898201e+01, 4.533901e+01, 5.169611e+01, 5.805321e+01, 6.436264e+01,
        7.062198e+01, 7.883422e+01, 8.909992e+01, 9.936521e+01, 1.091817e+02, 1.189586e+02,
        1.286959e+02, 1.429100e+02, 1.562600e+02, 1.696090e+02, 1.816190e+02, 1.930970e+02,
        2.032590e+02, 2.121500e+02, 2.187760e+02, 2.238980e+02, 2.243630e+02, 2.168650e+02,
        2.011920e+02, 1.769300e+02, 1.503930e+02, 1.278370e+02, 1.086630e+02, 9.236572e+01,
        7.851231e+01, 5.638791e+01, 4.017541e+01, 2.836781e+01, 1.979160e+01, 9.292942e+00,
        4.076571e+00, 1.650790e+00, 6.167791e-01, 2.113490e-01, 6.600001e-02, 1.000000e-02)
Bp <- c(1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
        8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
        7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
        5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
        2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
        6.960025e-03, 8.175413e-09, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00)

### Preallocate and loop over the particles
particles <- unique(part.ind)
nP        <- length(particles)
tU        <- rep(NA, nP)
xU        <- rep(NA, nP)
yU        <- rep(NA, nP)
zU        <- rep(NA, nP)
zK        <- rep(NA, nP)
eK        <- rep(NA, nP)
for (i in c(1:nP)){
try({
   # Get all data for this particle
   ind   <- part.ind == particles[i] 
   tT    <- part.time[ind]
   xT    <- part.lon[ind]
   yT    <- part.lat[ind]
   zT    <- part.agl[ind]
   # Index to use
   eI    <- length(tT)		# Last timestep
   # Find the ending location and index
   tU[i] <- tT[eI]
   xU[i] <- xT[eI]
   yU[i] <- yT[eI]
   zU[i] <- zT[eI]
   # Load the file
   #yyyy      <- as.numeric(strftime(run_time, paste0('%Y'), 'UTC'))
   #mm        <- as.numeric(strftime(run_time, paste0('%m'), 'UTC'))
   #dd        <- as.numeric(strftime(run_time, paste0('%d'), 'UTC'))
   #hh        <- as.numeric(strftime(run_time, paste0('%H'), 'UTC'))
   yyyy      <- as.numeric(strftime((part.time[ind])[eI], paste0('%Y'), 'UTC'))
   mm        <- as.numeric(strftime((part.time[ind])[eI], paste0('%m'), 'UTC'))
   dd        <- as.numeric(strftime((part.time[ind])[eI], paste0('%d'), 'UTC'))
   hh        <- as.numeric(strftime((part.time[ind])[eI], paste0('%H'), 'UTC'))
   nasaFile  <- sprintf("%s/2x25/%04i/%02i/%02i.nc",nasaDir,yyyy,mm,dd)
   ncfile    <- nc_open(nasaFile)
   nasaLON   <- ncvar_get(ncfile, varid="lon")
   nasaLAT   <- ncvar_get(ncfile, varid="lat")
   nasaPSURF <- ncvar_get(ncfile, varid="psurf")
   nasaLEV   <- ncvar_get(ncfile, varid="lev")
   nasaTIME  <- ncvar_get(ncfile, varid="time")
   nasaCO2   <- ncvar_get(ncfile, varid="CO2")
   nc_close(ncfile)
   # Find the lat/lon
   iT <- which.min(abs(hh    - nasaTIME))
   iX <- which.min(abs(xU[i] - nasaLON))
   iY <- which.min(abs(yU[i] - nasaLAT))
   # Get the actual concentration
   nasaPRES <- Ap + Bp * nasaPSURF[iX,iY,iT]
   nasaALT  <- -log(nasaPRES/nasaPSURF[iX,iY,iT]) * 7.4 # Assuming a scale height of 7.4 km
   iZ       <- which.min(abs(zU[i] - nasaALT))
   zK[i]    <- nasaCO2[iX,iY,iZ,iT] * 1e6
   # Debugging?
   if (debug){
      print(sprintf("NASA iter #%3i/%3i (%s): CO2 = %5.2fppm (%5.2fW,%5.2fN @ %5.2fkm)",i,nP,(part.time[ind])[eI],zK[i],xU[i],yU[i],zU[i]))
   }
})
}
# Store the NASA CO2
bkg_NASA <- zK
# Store the particle information
endLON <- xU
endLAT <- yU
endAGL <- zU

### ========================
### = Get the NOAA background
### ========================

### Preallocate and loop over the particles
particles <- unique(part.ind)
nP        <- length(particles)
tU        <- rep(NA, nP)
xU        <- rep(NA, nP)
yU        <- rep(NA, nP)
zK        <- rep(NA, nP)
eK        <- rep(NA, nP)
for (i in c(1:nP)){
try({
   # Get all data for this particle
   ind   <- part.ind == particles[i] 
   tT    <- part.time[ind]
   xT    <- part.lat[ind]
   yT    <- part.agl[ind]
   lons  <- part.lon[ind];lons[lons<0] <- lons[lons<0] + 360
   # Index to use
   eI    <- length(tT)		# Last timestep
   # Find the ending location and index
   tU[i] <- which.min(abs(tT[eI] - noaa.time))
   xU[i] <- xT[eI]
   yU[i] <- yT[eI]
   # Krig the data
   z     <- t(noaa.co2[,,tU[i]])
   e     <- t(noaa.err[,,tU[i]])
   zK[i] <- krig_fun(noaa.lat,noaa.alt,z,xU[i],yU[i],lx,ly)
   # Debugging?
   if (debug){
      print(sprintf("NOAA iter #%3i/%3i (%s): CO2 = %5.2fppm (%5.2fN,%5.2fW @ %5.2fkm)",i,nP,(part.time[ind])[eI],zK[i],xU[i],lons[eI],yU[i]))
   }
})
}
# Store the NOAA CO2
bkg_NOAA <- zK

### Return the background for each particle
out_list <- list("NASA"=bkg_NASA, "NOAA"=bkg_NOAA)
return(out_list)

}

### ====================================================
### =                    E N D                         =
### ====================================================

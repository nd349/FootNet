### STILT R Executable
### For documentation, see https://uataq.github.io/stilt/
### Ben Fasoli

### User inputs ------------------------------------------------------------------
project <- 'BARNETT'
stilt_wd <- file.path('/home/disk/p/nd349/nikhil.dadheech/barnette/STILT', project)
output_wd <- file.path(stilt_wd, 'out')
lib.loc <- .libPaths()[1]

### Load the necessary libraries

setwd(stilt_wd)
source('r/dependencies.r')
require(pracma)

### Parallel simulation settings
# simName <- 'serial'
simName <- 'hermes'
# Serial
n_nodes <- 1
n_cores <- 1
# savio (savio)
if (simName == 'hermes') {
    n_cores <- 32
    n_nodes <- 2
    slurm_options <- list(
     time      = '096:00:00',
     account   = 'nd349'
    )
}

slurm <- n_nodes > 1 | n_cores > 1
# print("slurm")
# print(slurm)
### Receptor locations
origin <- as.Date("1970-1-1")
ncfile <- nc_open(paste(stilt_wd,'/receptors/receptorFile.ncdf',sep=""))
fjul   <- ncvar_get(ncfile, varid="fjul") # Not used by STILT anymore
lati   <- ncvar_get(ncfile, varid="lat")
long   <- ncvar_get(ncfile, varid="lon")
zagl   <- ncvar_get(ncfile, varid="agl")
co2    <- ncvar_get(ncfile, varid="co2")
co2e   <- ncvar_get(ncfile, varid="co2e")
yr     <- ncvar_get(ncfile, varid="yr") # Construct julian dates based on yy-mm-dd-hh
mon    <- ncvar_get(ncfile, varid="mon")
day    <- ncvar_get(ncfile, varid="day")
hr     <- ncvar_get(ncfile, varid="hr")
nc_close(ncfile)
yyyymmddHH <- rep("yyyymmdd HH", length(co2))
for (i in 1:length(yyyymmddHH)) {
   yyyymmddHH[i] = sprintf("%04i/%02i/%02i %02i",yr[i],mon[i],day[i],hr[i])
}
run_times = as.POSIXct(yyyymmddHH, tz = 'UTC', format = "%Y/%m/%d %H")
#run_times = as.POSIXct(fjul + origin, tz = 'UTC')
# print(run_times)
### Filter for unique combinations
rID       <- strftime(run_times, paste0('%Y%m%d%H_', long, '_', lati, '_',zagl), 'UTC')
uID       <- !duplicated(rID)
rID       <- rID[uID]
run_times <- run_times[uID]
lati      <- lati[uID]
long      <- long[uID]
zagl      <- zagl[uID]
co2       <- co2[uID]
co2e      <- co2e[uID]

### Expand the run times, lats, and lons to form the unique receptors for each simulation
#receptors <- expand.grid(run_time = run_times, lati = lati, long = long,
#                         zagl = zagl, KEEP.OUT.ATTRS = F, stringsAsFactors = F)
receptors <- data.frame(run_time = run_times, lati = lati, long = long, zagl = zagl, co2 = co2, co2e = co2e)

# # Trim to 03/01/2019 and 03/05/2019

# Time slicing (Comment if need to be run over the whole data)
tID <- as.POSIXct('2013-10-19 00:00:00', tz = 'UTC') <= run_times & run_times < as.POSIXct('2013-10-19 23:00:00', tz = 'UTC')
receptors <- receptors[tID,]


# print(receptors)
### Meteorological data input
n_met_min       <- 1
# HRRR
met_directory   <- paste(stilt_wd,'/met_data/BarnettShale_2013/arl/MYJ_LSM/',sep="")
print("met_directory")
print(met_directory)
# met_file_format <- 'hysplit.%Y%m%d.%Hz.hrrra'
# WRF
#met_directory   <- paste(stilt_wd,'/met_data/wrf_arl/',sep="")
met_file_format <- 'd04.%Y%m%d.arl'

### Model control
n_hours    <- -72
numpar     <- 100
rm_dat     <- T
run_trajec <- T
run_foot   <- F 
overwrite  <- F
shuffle    <- T
timeout    <- 3600
varsiwant  <- c('time', 'indx', 'long', 'lati', 'zagl', 'sigw', 'tlgr', 'zsfc',
                'icdx', 'temp', 'samt', 'foot', 'shtf', 'tcld', 'dmas', 'dens',
                'rhfr', 'sphu', 'solw', 'lcld', 'zloc', 'dswf', 'wout', 'mlht',
                'rain', 'crai', 'pres')

### Footprint grid settings
hnf_plume <- T
projection <- '+proj=longlat'
smooth_factor <- 1
time_integrate <- F

### Define the footprint grid
# Resolution
#lon.res <- 1./120.         #resolution in degrees longitude
#lat.res <- 1./120.         #resolution in degrees latitude
# Number of grid cells
nLon    <- 175 
nLat    <- 217 
# Bounding region
#xlims <- c(-124.0, -121.0) # "hi-res" region in the Turner inventory
#ylims <- c(  37.0,   39.0) # "hi-res" region in the Turner inventory
xlims <- c(-98.9000015258789, -96.4000015258789) # Larger Area domain
ylims <- c(  31.5,   34.099998474121094) # Larger Area domain
# Make the grid
lons <- linspace(xlims[1],xlims[2],nLon)
lats <- linspace(ylims[1],ylims[2],nLat)
# Required params
xmn  <- xlims[1]
xmx  <- xlims[2]
ymn  <- ylims[1]
ymx  <- ylims[2]
nx   <- nLon
ny   <- nLat
xres <- lons[2] - lons[1]
yres <- lats[2] - lats[1]

### Transport and dispersion settings
conage      <- 48
cpack       <- 1
delt        <- 0
dxf         <- 1
dyf         <- 1
dzf         <- 0.1
emisshrs    <- 0.01
frhmax      <- 3
frhs        <- 1
frme        <- 0.1
frmr        <- 0
frts        <- 0.1
frvs        <- 0.1
hscale      <- 10800
ichem       <- 0
iconvect    <- 0
initd       <- 0
isot        <- 0
kbls        <- 1
kblt        <- 1
kdef        <- 1
khmax       <- 9999
kmix0       <- 250
kmixd       <- 3
kmsl        <- 0
kpuff       <- 0
krnd        <- 6
kspl        <- 1
kzmix       <- 1
maxdim      <- 1
maxpar      <- min(10000, numpar)
mgmin       <- 2000
ncycl       <- 0
ndump       <- 0
ninit       <- 1
nturb       <- 0
outdt       <- 0
outfrac     <- 0.9
p10f        <- 1
qcycle      <- 0
random      <- 1
splitf      <- 1
tkerd       <- 0.18
tkern       <- 0.18
tlfrac      <- 0.1
tratio      <- 0.9
tvmix       <- 1
veght       <- 0.5
vscale      <- 200
w_option    <- 0
zicontroltf <- 0
ziscale     <- rep(list(rep(0.8, 24)), nrow(receptors))
z_top       <- 25000

### Transport error settings
horcoruverr <- NA
siguverr    <- NA
tluverr     <- NA
zcoruverr   <- NA

horcorzierr <- NA
sigzierr    <- NA
tlzierr     <- NA


### Startup messages -------------------------------------------------------------
message('Initializing STILT')
message('Number of receptors: ', nrow(receptors))
message('Number of parallel threads: ', n_nodes * n_cores)


### Structure out directory ------------------------------------------------------
# Outputs are organized in three formats. by-id contains simulation files by
# unique simulation identifier. particles and footprints contain symbolic links
# to the particle trajectory and footprint files in by-id
if ( (run_trajec) && (overwrite) ) {
  system(paste0('rm -r ', output_wd, '/by-id'), ignore.stderr = T)
  system(paste0('rm -r ', output_wd, '/particles'), ignore.stderr = T)
  system(paste0('rm -r ', output_wd, '/footprints'), ignore.stderr = T)
}
for (d in c('by-id', 'particles', 'footprints')) {
  d <- file.path(output_wd, d)
  if (!file.exists(d))
    dir.create(d, recursive = T)
}


### Met path symlink -------------------------------------------------------------
# Auto symlink the meteorological data path to the working directory to
# eliminate issues with long (>80 char) paths in fortran. Note that this assumes
# that all meteorological data is found in the same directory.
if ((nchar(paste0(met_directory, met_file_format)) + 2) > 80) {
  #met_loc <- file.path(path.expand('~'), paste0('m', project))
  met_loc <- file.path(path.expand('~/met_symbolic_barnett'))
    print(1)
  #met_loc <- file.path(path.expand('~/scratch'), paste0('m', project))
  #system(paste0('rm ', met_loc), ignore.stderr = T)
  #if (!file.exists(met_loc)) invisible(file.symlink(met_directory, met_loc))
} else met_loc <- met_directory

print("met_loc:")
print(met_loc)
### Run trajectory simulations ---------------------------------------------------
# Gather varsiwant into a single character string and fork the process to apply
# simulation_step() to each receptor across n_cores and n_nodes
validate_varsiwant(varsiwant)
if (!is.null(varsiwant[1]))
  varsiwant <- paste(varsiwant, collapse = '/')
output <- stilt_apply(X = 1:nrow(receptors), FUN = simulation_step,
                      slurm = slurm, slurm_options = slurm_options,
                      n_cores = n_cores, n_nodes = n_nodes, rm_dat = rm_dat,
                      conage = conage, cpack = cpack, delt = delt,
                      emisshrs = emisshrs, frhmax = frhmax, frhs = frhs,
                      frme = frme, frmr = frmr, frts = frts, frvs = frvs,
                      hnf_plume = hnf_plume, horcoruverr = horcoruverr,
                      horcorzierr = horcorzierr, ichem = ichem,
                      iconvect = iconvect, initd = initd, isot = isot,
                      kbls = kbls, kblt = kblt, kdef = kdef, khmax = khmax,
                      kmix0 = kmix0, kmixd = kmixd, kmsl = kmsl, kpuff = kpuff,
                      krnd = krnd, kspl = kspl, kzmix = kzmix, maxdim = maxdim,
                      maxpar = maxpar, lib.loc = lib.loc,
                      met_file_format = met_file_format, met_loc = met_loc,
                      mgmin = mgmin, n_hours = n_hours, n_met_min = n_met_min,
                      ncycl = ncycl, ndump = ndump, ninit = ninit,
                      nturb = nturb, numpar = numpar, outdt = outdt,
                      outfrac = outfrac, output_wd = output_wd, p10f = p10f,
                      projection = projection, qcycle = qcycle, overwrite = overwrite,
                      r_run_time = receptors$run_time, r_lati = receptors$lati,
                      r_long = receptors$long, r_zagl = receptors$zagl,
                      random = random, run_trajec = run_trajec, run_foot = run_foot,
                      shuffle = shuffle, siguverr = siguverr, sigzierr = sigzierr,
                      smooth_factor = smooth_factor, splitf = splitf,
                      stilt_wd = stilt_wd, time_integrate = time_integrate,
                      timeout = timeout, tkerd = tkerd, tkern = tkern,
                      tlfrac = tlfrac, tluverr = tluverr, tlzierr = tlzierr,
                      tratio = tratio, tvmix = tvmix, varsiwant = varsiwant,
                      veght = veght, vscale = vscale, w_option = w_option,
                      xmn = xmn, xmx = xmx, xres = xres, ymn = ymn, ymx = ymx,
                      yres = yres, nx = nx, ny = ny, zicontroltf = zicontroltf, ziscale = ziscale,
                      z_top = z_top, zcoruverr = zcoruverr)

q('no')

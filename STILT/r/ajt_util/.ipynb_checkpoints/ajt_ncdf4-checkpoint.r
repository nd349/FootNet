### ====================================================
### = ajt_ncdf4.r
### = Alex Turner
### = 07/22/2014
### =---------------------------------------------------
### = NOTES:
### = ( 1) A function to write the obs data to ncdf4
### ====================================================

### Load the necessary libraries
require(ncdf4)

### Function to write out the netCDF file
write_ncdf4 <- function(fname, obs_yr, obs_mon, obs_day, obs_hr, obs_jul, obs_lon, obs_lat, obs_agl, obs_co2, obs_err, bkg_val_NOAA, bkg_err_NOAA, bkg_val_NASA, bkg_err_NASA, lon_vec, lat_vec, ftimes, foot) {
   out <- tryCatch(
       {
          dimINFO  <- ncdim_def("info","na",1)
          dimTIME  <- ncdim_def("time","seconds since 1970-01-01 00:00:00Z",ftimes)
	  dimLON   <- ncdim_def("lon","degrees_east",lon_vec)
	  dimLAT   <- ncdim_def("lat","degrees_north",lat_vec)
          # Define the objects
	  var_foot <- ncvar_def("foot", "ppm/(umol*m-2*s-1)", compression=9,
             list(dimLON,dimLAT,dimTIME), -999, longname="stilt surface influence footprint")
          var_co2 <- ncvar_def("co2", "ppm",
             dimINFO, -999, longname="dry air mole mixing ratio at the receptor")
          var_co2_err <- ncvar_def("co2_err", "ppm",
             dimINFO, -999, longname="Error in the dry air mole mixing ratio at the receptor")
          var_bkg_NOAA <- ncvar_def("bkg_co2_NOAA", "ppm",
             dimINFO, -999, longname="dry air mole mixing ratio at the NOAA curtain")
          var_bkg_err_NOAA <- ncvar_def("bkg_err_NOAA", "ppm",
             dimINFO, -999, longname="Error in the dry air mole mixing ratio at the NOAA curtain")
          var_bkg_NASA <- ncvar_def("bkg_co2_NASA", "ppm",
             dimINFO, -999, longname="dry air mole mixing ratio from the NASA CMS run")
          var_bkg_err_NASA <- ncvar_def("bkg_err_NASA", "ppm",
             dimINFO, -999, longname="Error in the dry air mole mixing ratio from the NASA CMS run")
          var_jul <- ncvar_def("jul",  "seconds since 1970-01-01 00:00:00Z", dimINFO,
             -999, longname="julian date")
          var_yr  <- ncvar_def("yr",  "yyyy", dimINFO,
             -999, longname="stilt back trajectory start time")
          var_mon <- ncvar_def("mon", "mm",   dimINFO,
             -999, longname="stilt back trajectory start time")
          var_day <- ncvar_def("day", "dd",   dimINFO,
             -999, longname="stilt back trajectory start time")
          var_hr  <- ncvar_def("hr",  "hh",   dimINFO,
             -999, longname="stilt back trajectory start time")
          var_obs_lat <- ncvar_def("obs_lat", "degrees_north",   dimINFO,
             -999, longname="stilt back trajectory start longitude")
          var_obs_lon <- ncvar_def("obs_lon", "degrees_east",   dimINFO,
             -999, longname="stilt back trajectory start latitude")
          var_obs_agl <- ncvar_def("obs_agl", "m AGL",   dimINFO,
             -999, longname="stilt back trajectory start altitude")
          # Begin writing
          ncid <- nc_create(fname, force_v4=TRUE,
             list(var_co2,          var_co2_err,  var_bkg_NOAA,
                  var_bkg_err_NOAA, var_bkg_NASA, var_bkg_err_NASA,
                  var_yr,           var_mon,      var_day,
                  var_hr,           var_jul,      var_obs_lat, 
                  var_obs_lon,      var_obs_agl,  var_foot))
          nc_close(ncid)
          ncid <- nc_open(fname,write=TRUE)
          # Add the variables
	  ncvar_put(ncid, var_yr,           obs_yr)
	  ncvar_put(ncid, var_mon,          obs_mon)
	  ncvar_put(ncid, var_day,          obs_day)
	  ncvar_put(ncid, var_hr,           obs_hr)
	  ncvar_put(ncid, var_jul,          obs_jul)
	  ncvar_put(ncid, var_obs_lon,      obs_lon)
	  ncvar_put(ncid, var_obs_lat,      obs_lat)
	  ncvar_put(ncid, var_obs_agl,      obs_agl)
          ncvar_put(ncid, var_foot,         foot)
	  ncvar_put(ncid, var_co2,          obs_co2)
	  ncvar_put(ncid, var_co2_err,      obs_err)
	  ncvar_put(ncid, var_bkg_NOAA,     bkg_val_NOAA)
	  ncvar_put(ncid, var_bkg_err_NOAA, bkg_err_NOAA)
	  ncvar_put(ncid, var_bkg_NASA,     bkg_val_NASA)
	  ncvar_put(ncid, var_bkg_err_NASA, bkg_err_NASA)
	  nc_close(ncid)
          # If successful we'll return 1
          return(1)
       }, error=function(cond) {
            message("Something went wrong, here's message:")
            message(cond)
            # If we get an error we'll return 0
            return(0)
       }, warning=function(cond) {
            message("Warning!:")
            message(cond)
            # If we get a warning we'll return null
            return(NULL)
       }
       )
   return(out) # 1 = success, 0 = error, NULL = warning
}

### ====================================================
### =                    E N D                         =
### ====================================================

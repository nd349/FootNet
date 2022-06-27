%%% =======================================================================
%%% = Generate_Receptors.m
%%% = Alex Turner (AJT)
%%% = Created on 06/06/2014
%%% =----------------------------------------------------------------------
%%% = NOTES
%%% =  ( 1): Makes a receptor file from the BeACON data.
%%% =======================================================================

function [ ] = Generate_Receptors( )
   
%%% Clear old data
clf
close all
clear all
clc

%%% Define some paths
intDir = pwd;
datDir = sprintf('%s/data/csv',intDir);
outDir = sprintf('%s/data/netcdf',intDir);

%%% File format
fspec = '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f';

%%% Plot options
pOPTS = {'FontName','Helvetica','FontSize',16,'FontWeight','Bold'};


%%% =======================================================================
%%% ( 1) :: BEGIN READING THE DATA
%%% =======================================================================

%%% Lat/Lon/Elevation for each site:
fid    = fopen(sprintf('%s/data/BEACON_sites.csv',intDir));
sites  = textscan(fid, '%f%s%f%f%f','Delimiter',',','Headerlines',1);
nSites = length(sites{1}(:));
fclose(fid);

%%% Loop over each site
for i = 1:nSites

    %%% Do we have this file?
    outname = sprintf('%s/%s-beacon.ncdf',outDir,sites{2}{i});
    inname  = sprintf('%s/BAD_%s-beacon.ncdf.~bad~',outDir,sites{2}{i});
    if (exist(outname, 'file') == 0)
        if (exist(inname, 'file') == 2)
            
            %%% Parameters
            param = {'serial_date','year','month','day','hour','minute','second',...
                     'CO2_viasala','CO2_error_viasala',...
                     'XCO2','XCO2_error',...
                     'XCO2_TP','XCO2_error_TP',...
                     'XCO2_TPV','XCO2_error_TPV',...
                     'error_1','error_2','error_3','error_4','error_5','error_6','error_7'};
            
            %%% Loop over the parameters
            data_write = cell(22,1);
            for j = 1:length(param)
                data_write{j} = ReadNetCDFVar( inname, param{j} );
            end
            
            %%% Write the new file
            WriteBEACON( outname, data_write, sites{2}{i}, sites{3}(i), sites{4}(i), sites{5}(i) )
            
        end
    end
end


%%% =======================================================================
%%% ( 3) :: PLOT ALL OF THE BEACON DATA
%%% =======================================================================

%%% List all the files with data
files  = dir(sprintf('%s/*.ncdf',outDir));
nFiles = length(files);

%%% Read all the files
data = cell(nFiles,7);
for i = 1:nFiles
    site_name = strsplit(files(i).name, '-');
    data{i,1} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'serial_date' );
    data{i,2} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'XCO2_TPV' );
    data{i,3} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'XCO2_error_TPV' );
    data{i,4} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'lat' );
    data{i,5} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'lon' );
    data{i,6} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'agl' );
    data{i,7} = site_name{1};
end
    
%%% Plot all the data
h_beacon = figure();
hold on
grid on
box on
for i = 1:nFiles
    x_dat = data{i,1}(:);
    y_dat = data{i,2}(:);
    ind   = 100 < y_dat & y_dat < 1500;
    plot(x_dat(ind),y_dat(ind),'k.','MarkerSize',2)
end
datetick('x','mm/yyyy')
xlim([datenum('01/2013','mm/yyyy'),datenum('01/2014','mm/yyyy')])


%%% Plot all the data
XTickLabels = {'08/01/13','08/11/13','08/21/13','08/31/13','09/10/13','09/20/13','09/30/13','10/10/13','10/20/13','10/31/13'};
XTicks      = zeros(length(XTickLabels),1);
for i = 1:length(XTicks)
    XTicks(i)      = datenum(XTickLabels{i},'mm/dd/yy');
    XTickLabels{i} = datestr(XTicks(i),'mmm dd');
end
h_beacon_sub = figure();
hold on
grid on
box on
xLim = [datenum('09/01/2013','mm/dd/yyyy'),datenum('11/01/2013','mm/dd/yyyy')];
cmap = jet(nFiles);
x_full = [];
y_full = [];
for i = 1:nFiles
    Lval  = (i - 1)/(nFiles - 1) * (length(cmap) - 1) + 1;
    col   = interp1(1:length(cmap),cmap,Lval);
    x_dat = data{i,1}(:);
    y_dat = data{i,2}(:);
    ind   = 100     <  y_dat & y_dat <  1500    & ...
            xLim(1) <= x_dat & x_dat <  xLim(2);
    plot(x_dat(ind),y_dat(ind),'k.','MarkerSize',2,'MarkerEdgeColor',col)
    x_full = [x_full;x_dat(ind)];
    y_full = [y_full;y_dat(ind)];
end
datetick('x','mm/yyyy')
ylim([350 550])
xlim(xLim)
ylabel('XCO_2 [ppm]',pOPTS{:})
set(gca,'XTick',XTicks,'XTickLabel',XTickLabels)
set(gca,'LineWidth',2)
set(gca,pOPTS{:})
set(gca,'XMinorTick','on','YMinorTick','on')

%%% Save the data and plots?
if false
    PaperSize1 = [20 5];
    PaperSize2 = [0 0 PaperSize1];
    set(h_beacon_sub, 'PaperUnits', 'inches');
    set(h_beacon_sub, 'PaperSize', PaperSize1);
    set(h_beacon_sub, 'PaperPosition', PaperSize2);
    
    print(h_beacon_sub,'-depsc2',sprintf('%s/output/BEACON_ts_use.eps',intDir));
end

%%% =======================================================================
%%% ( 3) :: SAVE ALL THE BEACON DATA FOR STILT
%%% =======================================================================

%%% Read all the files
data = cell(nFiles,7);
for i = 1:nFiles
    site_name = strsplit(files(i).name, '-');
    data{i,1} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'serial_date' );
    data{i,2} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'XCO2_TPV' );
    data{i,3} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'XCO2_error_TPV' );
    data{i,4} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'lat' );
    data{i,5} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'lon' );
    data{i,6} = ReadNetCDFVar( sprintf('%s/%s',outDir,files(i).name), 'agl' );
    data{i,7} = site_name{1};
end

%%% Get the data in our time period
jDate    = datenum('01/01/1960','mm/dd/yyyy'); % Start of julian date in R
xLim     = [datenum('01/01/2013','mm/dd/yyyy'),datenum('01/01/2014','mm/dd/yyyy')];
fjul     = [];
lat      = [];
lon      = [];
agl      = [];
XCO2     = [];
XCO2_err = [];
for i = 1:nFiles
    ind      = xLim(1) <= data{i,1} & data{i,1} < xLim(2);
    fjul     = [fjul;data{i,1}(ind)-jDate];
    lat      = [lat;ones(sum(ind),1)*data{i,4}(1)];
    lon      = [lon;ones(sum(ind),1)*data{i,5}(1)];
    agl      = [agl;ones(sum(ind),1)*data{i,6}(1)];
    XCO2     = [XCO2;data{i,2}(ind)];
    XCO2_err = [XCO2_err;data{i,3}(ind)];
end

%%% Write the receptor file
outname = sprintf('%s/../receptor_file.ncdf',outDir);
WriteSTILT( outname, fjul, lat, lon, agl, XCO2, XCO2_err )

end


%%% =======================================================================
%%% = ReadNetCDFVar
%%% = Alex Turner (AJT)
%%% = Created on 03/22/2013
%%% =----------------------------------------------------------------------
%%% = NOTES
%%% =  ( 1): Reads a single variable from a NetCDF file.
%%% =----------------------------------------------------------------------
%%% = INPUTS
%%% =  ( 1): FileName = Filename to read from.
%%% =  ( 2): VarName  = Variable name to read.
%%% =----------------------------------------------------------------------
%%% = OUTPUTS
%%% =  ( 1): data     = Data from the variable.
%%% =======================================================================

function [ data ] = ReadNetCDFVar( FileName, VarName )

%%% Read the NetCDF Data
ncid  = netcdf.open(FileName,'NOWRITE');    % Get an id for the file
varid = netcdf.inqVarID(ncid,VarName);      % Get the variable id
data  = netcdf.getVar(ncid,varid,'double'); % Read in double precision
netcdf.close(ncid)                          % Close the file

end


%%% =======================================================================
%%% = WriteBEACON
%%% = Alex Turner (AJT)
%%% = Created on 06/06/2014
%%% =======================================================================
% 
% The columns of those files (starting with zero) are:
% 
% 0: timestamp
% 1: Vaisala measured CO2 [ppm]
% 2: Error corresponding to value in column 1, includes statistical and systematic error.
% 3: Dry air mole fraction assuming only the temperature correction is important
% 4: Propagated total error (standard deviation) corresponding to column 3
% 5-6: Analog to columns 3-4, assuming only the joint temperature and pressure correction is important
% 7-8: Analog to columns 3-4, corrected for temperature, pressure, and water
% 9-15: Contributions to the total variance leading to calculated standard deviation in column 8; terms 1-7 in equation 46 of the calibration document (attached).

function [  ] = WriteBEACON( FileName, dat, site, lat, lon, elevation )

%%% Define some dimensions
nObs   = length(dat{1}(:));
nParam = 22;

%%% Define the file names
param = {'serial_date','year','month','day','hour','minute','second',...
         'CO2_viasala','CO2_error_viasala',...
         'XCO2','XCO2_error',...
         'XCO2_TP','XCO2_error_TP',...
         'XCO2_TPV','XCO2_error_TPV',...
         'error_1','error_2','error_3','error_4','error_5','error_6','error_7'};

%%% Add the single parameters
% Lat
nccreate(FileName,'lat','Dimensions',{'about',1},'Format','classic');
ncwrite(FileName,'lat',lat);
% Lon
nccreate(FileName,'lon','Dimensions',{'about',1},'Format','classic');
ncwrite(FileName,'lon',lon);
% Elevation
nccreate(FileName,'elev','Dimensions',{'about',1},'Format','classic');
ncwrite(FileName,'elev',elevation);

%%% Loop over the parameters to add
for i = 1:nParam;
    nccreate(FileName,param{i},'Dimensions',{'nObs',nObs},'Format','classic');
    ncwrite(FileName,param{i},dat{i}(:));
end

%%% Attributes
ncwriteatt(FileName,'/','site',site);
ncwriteatt(FileName,'/','missing_data','-999');
ncwriteatt(FileName,'/','creation_time',datestr(now));
ncdisp(FileName);

end


%%% =======================================================================
%%% = WriteSTILT
%%% = Alex Turner (AJT)
%%% = Created on 06/18/2014
%%% =----------------------------------------------------------------------
%%% = NOTES
%%% =  ( 1): Writes the clusters to a text file.
%%% =----------------------------------------------------------------------
%%% = INPUTS
%%% =  ( 1): FileName = Filename to write to.
%%% =  ( 2): Clusters = Variable to write.
%%% =  ( 3): mu       = Variable to write.
%%% =  ( 4): cov      = Variable to write.
%%% =  ( 5): phi      = Variable to write.
%%% =  ( 6): w        = Variable to write.
%%% =  ( 7): X        = Longitude.
%%% =  ( 8): Y        = Latitude.
%%% =----------------------------------------------------------------------
%%% = OUTPUTS
%%% =  N/A
%%% =======================================================================

function [  ] = WriteSTILT( FileName, fjul, lat, lon, agl, XCO2, XCO2_err )

%%% Define some dimensions
nObs = length(lat);

%%% Add the parameters
% fjul
nccreate(FileName,'fjul','Dimensions',{'obs',nObs},'Format','classic');
ncwrite(FileName,'fjul',fjul);
% lat
nccreate(FileName,'lat','Dimensions',{'obs',nObs},'Format','classic');
ncwrite(FileName,'lat',lat);
% lon
nccreate(FileName,'lon','Dimensions',{'obs',nObs},'Format','classic');
ncwrite(FileName,'lon',lon);
% agl
nccreate(FileName,'agl','Dimensions',{'obs',nObs},'Format','classic');
ncwrite(FileName,'agl',agl);
% XCO2
nccreate(FileName,'XCO2','Dimensions',{'obs',nObs},'Format','classic');
ncwrite(FileName,'XCO2',XCO2);
% XCO2_err
nccreate(FileName,'XCO2_err','Dimensions',{'obs',nObs},'Format','classic');
ncwrite(FileName,'XCO2_err',XCO2_err);


%%% Attributes
ncwriteatt(FileName,'/','creation_time',datestr(now));
ncdisp(FileName);

end


%%% =======================================================================
%%% END
%%% =======================================================================

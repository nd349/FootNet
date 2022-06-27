%%% =======================================================================
%%% = CompressFootprints.m
%%% = Alex Turner (AJT)
%%% = Created on 03/04/2020
%%% =----------------------------------------------------------------------
%%% = NOTES
%%% =  ( 1): Compress the footprints
%%% =======================================================================

function [ ] = CompressFootprints( )

%clf; close all; clear all; clc;

%%% Define the directories data
baseDir = pwd;
%footDir = sprintf('%s/out/footprints/',baseDir);
footDir = sprintf('%s/out/obs/',baseDir);
compDir = sprintf('%s/out/compressedFoot/',baseDir);

print(footDir, compDir)

%%% Are we overwriting?
overwrite = false;

%%% Get a list of all the files
files = dir(sprintf('%s/*.nc',footDir));


%%% =======================================================================
%%% Begin compressing the footprints
%%% =======================================================================

%%% Save the file that we're currently looking at
%delete ~/matlab_crash_*
%delFile = './delFile.txt';
%if exist(delFile,'file') ==2
%    fid = fopen(delFile);
%    currFile = textscan(fid,'%s');
%    currFile = char(currFile{1});
%    fclose(fid);
%    % Delete files
%    fprintf(' *** Deleting %s\n',currFile)
%    delete(sprintf('%s/out/by-id/*/%s',baseDir,currFile))
%    delete(sprintf('%s/out/footprints/%s',baseDir,currFile))
%    delete(delFile)
%end

%%% Compute the actual SIF
fprintf('\n  *** COMPRESS THE FOOTPRINTS ***\n')
for i = 1:length(files)

    %%% Diagnostic
    clear lat lon jdate foot footC
    fprintf('%6i) %s: ',i,files(i).name);tic;

    %%% Define the filenames
    footFile = sprintf('%s/%s',footDir,files(i).name);
    outFile  = sprintf('%s/%s',compDir,files(i).name);

    %%% Does it exist and/or are we overwriting it?
    if (exist(outFile,'file') ~=2 || overwrite)
        try % Wrap it in a try-catch
            % Save the file name that we're reading
            %fid = fopen(delFile,'wt');
            %fprintf(fid,files(i).name);
            %fclose(fid);
            % Get the data
            lat   = ncread(footFile,'lat');
            lon   = ncread(footFile,'lon');
            jdate = ncread(footFile,'time');
            foot  = ncread(footFile,'foot');
            footC = nansum(foot,3);
            % Save the result
            WriteData(outFile,lon,lat,single(jdate),single(footC))
            % Remove the delete file
            %delete(delFile);
        catch
            %delete(sprintf('%s/out/by-id/*/%s',baseDir,files(i).name))
            %delete(footFile)
            fprintf(' *** ERROR WITH %s ***',files(i).name)
        end
    end
    toc

end

end


function [  ] = WriteData( FileName, lon, lat, jdate, foot )

%%% Create the netcdf file
ncid = netcdf.create(FileName,'NETCDF4');
% Define the dimensions
latdimid = netcdf.defDim(ncid,'lat',length(lat));
londimid = netcdf.defDim(ncid,'lon',length(lon));
timdimid = netcdf.defDim(ncid,'time',length(jdate));

%%% Define the variables
lonid  = netcdf.defVar(ncid,'lon','double',[londimid]);
latid  = netcdf.defVar(ncid,'lat','double',[latdimid]);
timid  = netcdf.defVar(ncid,'time','NC_FLOAT',[timdimid]);
footid = netcdf.defVar(ncid,'foot','NC_FLOAT',[londimid,latdimid]);

%%% Deflate the large variables
netcdf.defVarDeflate(ncid,footid,true,true,9);

%%% Add attributes
netcdf.putAtt(ncid,latid,'Unit','degrees')
netcdf.putAtt(ncid,lonid,'Unit','degrees')
netcdf.putAtt(ncid,timid,'Unit','seconds since 1970-01-01 00:00:00Z')
netcdf.putAtt(ncid,footid,'Unit','ppm/(umol m2 s)')
netcdf.putAtt(ncid,netcdf.getConstant('GLOBAL'),'creation_date',datestr(now));

%%% End define mode
netcdf.endDef(ncid);

%%% Add the variables
netcdf.putVar(ncid,lonid,lon);
netcdf.putVar(ncid,latid,lat);
netcdf.putVar(ncid,timid,jdate);
netcdf.putVar(ncid,footid,foot);

%%% Close the netcdf file
netcdf.close(ncid)

end


%%% =======================================================================
%%% END
%%% =======================================================================

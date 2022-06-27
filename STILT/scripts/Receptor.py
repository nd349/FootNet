import os
import geopy, random, time
import geopy.distance
import pandas as pd
import netCDF4 as nc
from tqdm import tqdm

dx = 12
dy = 12
dz = 1
horizontal_distance = 10 # km
vertical_distance = 3 # m
startLat = 32.5
startLong = -98.0
startHeight = 9
outputFile = "BARNETT/receptors/receptorFile"
# outputFile = 'sampleReceptors'
fileType = "ncdf"

start_date = '2013-10-01 00:00:00'
end_date = '2013-10-31 23:00:00'
timezone = 'UTC'
time_freq = '1H'


class Receptors():
    def __init__(self):
        # self.startLat = startLat
        # self.startLong = startLong
        # self.startHeight = startHeight
        # self.xInterval = dx
        # self.yInterval = dy
        # self.zInterval = dz
        self.outputFile = outputFile
        self.fileType = fileType
        self.distance = horizontal_distance
        self.vertical_distance = vertical_distance
        self.start_date = start_date
        self.end_date = end_date
        self.timezone = timezone
        self.time_freq = time_freq
        self.timeseries = self.record_time()

    def getLatLong(self, start, distance, bearing):
        d = geopy.distance.distance(kilometers=distance)
        return d.destination(point=start, bearing=bearing)

    def getConc(self):
        # return random.random()
        return 0
    
    def record_time(self):
        timeseries = list(pd.date_range(start=self.start_date, end=self.end_date, tz=self.timezone, freq=self.time_freq))
        return [[entry.year, entry.month, entry.day, entry.hour] for entry in timeseries]
        
    def getLineReceptors(self, dx, dy, dz, startLat, startLong, startHeight):
        finalList = []
        lati = startLat
        longi = startLong
        heighti = startHeight
        start = geopy.Point(lati, longi)
        for x in tqdm(range(dx)):
            for y in range(dy):
                nextpoint = self.getLatLong(start, y*self.distance, 90)
                # finalList += [[nextpoint.latitude, nextpoint.longitude, z*10, self.getConc(), 0] for z in range(startHeight, startHeight+dz)]
                
                for date_info in self.timeseries:
                    temp = [[nextpoint.latitude, nextpoint.longitude, idx*vertical_distance+startHeight, self.getConc(), 0]+date_info for idx, z in enumerate(range(startHeight, startHeight+dz))]
                    finalList += temp
                
            # start = geopy.Point(lati, longi)
            start = self.getLatLong(start, self.distance, 0)
            for date_info in self.timeseries:
                temp = [[start.latitude, start.longitude, idx*vertical_distance+startHeight, self.getConc(), 0]+date_info for idx, z in enumerate(range(startHeight, startHeight+dz))]
                finalList += temp
        df = self.writeReceptors(finalList)
        return finalList

    def writeReceptors(self, data):
        # import pdb; pdb.set_trace()
        # df = pd.DataFrame(data, columns=['lati', 'long', 'VerticalHeight', 'co2', 'fjul'])
        df = pd.DataFrame(data, columns=['lati', 'long', 'VerticalHeight', 'co2', 'fjul', 'yr', 'mon', 'day', 'hr'])
        df['co2e'] = df['co2']
        df = df.sample(frac=1).reset_index(drop=True)
        print("Shape:", df.shape)
        if self.fileType == "nc" or self.fileType =="ncdf":
            df1 = df.to_xarray()
            try:
                os.system('rm '+self.outputFile+"."+self.fileType)
                print("writing the netcdf file")
                out_nc = nc.Dataset(f"{self.outputFile}.ncdf", 'w', format='NETCDF4')
                out_nc.createDimension("id", df.shape[0])
                out_nc.createVariable("lat", "f8", ("id"))[:] = df['lati']
                out_nc.createVariable("lon", "f8", ("id"))[:] = df['long']
                out_nc.createVariable("agl", "f8", ("id"))[:] = df['VerticalHeight']
                out_nc.createVariable("co2", "f8", ("id"))[:] = df['co2']
                out_nc.createVariable("co2e", "f8", ("id"))[:] = df['co2e']
                out_nc.createVariable("yr", "f8", ("id"))[:] = df['yr']
                out_nc.createVariable("mon", "f8", ("id"))[:] = df['mon']
                out_nc.createVariable("day", "f8", ("id"))[:] = df['day']
                out_nc.createVariable("hr", "f8", ("id"))[:] = df['hr']
                out_nc.createVariable("fjul", "f8", ("id"))[:] = df['fjul']
                out_nc.close()
            except Exception as e:
                print(e)
        else:
            df.to_csv(self.outputFile+".csv", index=False)
        return df


# data = getLineReceptors(dx, dy, dz, startLat, startLong, startHeight)

# df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'VerticalHeight', 'Concentration', 'Timestamp'])
# df.to_csv("SampleReceptors.tsv", sep='\t', index=False)

if __name__ == '__main__':
    receptor = Receptors()
    data = receptor.getLineReceptors(dx, dy, dz, startLat, startLong, startHeight)
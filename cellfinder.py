#!/usr/bin/env python3

# Copyright © 2017 Christopher N. Lawrence <lordsutch@gmail.com>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np

import glob
import pyproj
import sys
import folium
import math

from scipy.optimize import curve_fit, minimize

from haversine import haversine

ECEF = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
LLA = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

def LLAtoECEF(lat, lon, alt):
    return pyproj.transform(LLA, ECEF, lon, lat, alt, radians=False)

def ECEFtoLLA(x, y, z):
    lon, lat, alt = pyproj.transform(ECEF, LLA, x, y, z, radians=False)
    return (lat, lon, alt)

def EarthRadiusAtLatitude(lat):
    rlat = np.deg2rad(lat)
    
    # Refine estimate - stolen from Wikipedia
    a = np.float64(6378137.0)
    b = np.float64(6356752.3)
    
    rad = np.sqrt(((a*a*np.cos(rlat))**2 + (b*b*np.sin(rlat))**2) /
                  ((a*np.cos(rlat))**2 + (b*np.sin(rlat))**2))
    return rad

def find_tower_svd(readings, returnAlt=False):
    rad = EarthRadiusAtLatitude(readings['latitude'].mean())
    #print(rad)

    dists = readings['estDistance'].values
    #print(dists)

    x, y, z = LLAtoECEF(readings['latitude'].values,
                        readings['longitude'].values,
                        readings['altitude'].values)

    A = np.array([-2*x, -2*y, -2*z, rad*rad + x*x + y*y + z*z - dists*dists]).T

    (_, _, v) = np.linalg.svd(A)
    w = v[3,:]
    result = w/w[3]
    #print(result)

    lat, lon, alt = ECEFtoLLA(result[0], result[1], result[2])

    # Check for wrong solution
    dist = haversine((lat, lon), readings[['latitude', 'longitude']].iloc[0,:])
    if dist > 3000:
        print(readings)
        print(dist, lat, lon)
        #print(result)
        lat, lon = 90-lat, (lon-180)
        #print(LLAtoECEF(lat, lon, alt))
        print(lat, lon)
    
    if returnAlt:
        return (lat, lon, alt)
    else:
        return (lat, lon)

def find_startpos(readings):
    #return find_tower_svd(readings)
    
    row = readings['estDistance'].idxmin()
    #print(row)
    dat = readings.loc[row,:]
    return dat[['latitude', 'longitude']]
    
def distance(x, *p):
    #print(p, x[0])
    ystar = [haversine(xi, p) for xi in x]
    #print(np.mean(ystar))
    return ystar

def find_tower_curve(readings):
    #startpos = find_tower_leastsquares(readings)
    #startpos = (np.mean(readings['latitude'].values),
    #            np.mean(readings['longitude'].values))
    startpos = find_startpos(readings)
    #print(startpos)

    errors = [0.14985*2]*readings.shape[0]
    
    result, covm = curve_fit(distance,
                             readings[['latitude', 'longitude']].values,
                             readings['estDistance'].values/1000.0,
                             p0=(startpos[0], startpos[1]),
                             sigma=errors, absolute_sigma=True,
                             bounds=((-90, -180), (90, 180)))
    return result

def mse(x, locations, distances):
    mse = 0.0
    for location, distance in zip(locations, distances):
        dist = haversine(x, location)
        mse += (dist - distance)**2
    return mse/len(distances)

def find_tower(readings):
    startpos = find_startpos(readings)
    #print(startpos)

    result = minimize(mse, startpos,
                      args=(readings[['latitude', 'longitude']].values,
                            readings['estDistance'].values/1000.0),
                      method='L-BFGS-B',
                      bounds=((-90, 90), (-180, 180)),
                      options={'ftol': 1e-5, 'maxiter': 1e7})
    #print(result.x)

    dist = haversine(startpos, result.x)
    if dist > 100:
        print('* SVD error')
        print(startpos, result.x, dist)
        print(readings)
    
    return result.x

def pointAtDistanceAndBearing(row):
    lat1, lon1 = np.deg2rad(row.startpos)
    bearing = np.deg2rad(row.bearing)

    rad = EarthRadiusAtLatitude(lat1)

    dr = row.distance/rad
    lat2 = math.asin(math.sin(lat1) * math.cos(dr) +
                     math.cos(lat1) * math.sin(dr) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(dr) * math.cos(lat1),
                             math.cos(dr) - math.sin(lat1)*math.sin(lat2))
    return pd.Series([np.rad2deg(lat2), np.rad2deg(lon2)])

def threshold_round(a, clip):
    return np.round(a / clip)*clip

def test_find_tower():
    pos1 = np.array([32.64504, -83.70882])
    alt = 120

    N = 5000
    angle_range = 5 # All points within same angle_range degrees

    dists = np.random.random(N)*35000
    estdists = threshold_round(dists, 149.85)

    #bearings = np.random.random(N)*360

    bearings = np.random.random(N)*angle_range + np.random.random(1)*360

    Adict = {'startpos' : (pos1,) * N,
             'distance' : dists,
             'bearing' : bearings}
    A = pd.DataFrame(Adict)

    coords = A.apply(pointAtDistanceAndBearing, axis=1)
    #print(coords)
    
    Bdict = {'latitude' : coords.iloc[:,0], 'longitude' : coords.iloc[:,1],
             'altitude' : alt-100+np.random.random(N)*200,
             'estDistance' : estdists}
    B = pd.DataFrame(Bdict)
    
    guess = find_tower(B)
    print(guess-pos1)

    guess = find_tower_curve(B)
    print(guess-pos1)

    guess = find_tower_svd(B, returnAlt=True)
    print(guess)
    print(guess[:2]-pos1)
    print(guess[2]-alt)

def check_sanity(guess, readings):
    coords = readings[['latitude', 'longitude']]
    dists = coords.apply(lambda row: haversine(row, guess), axis=1)

    resid = (dists - readings.estDistance/1000.0)
    errors = (np.abs(resid) > 50).any() # Anything over 50 km off
    
    if errors:
        print(readings)
        print(guess)
        print(resid)
        sys.exit(1)
    
icon_color = {25: 'red', 41: 'lightred', 26: 'darkred',
              17: 'lightgreen', 12: 'green', 2: 'green',
              5: 'purple'}

band_color = {25: 'red', 41: '#FFC0CB', 26: 'maroon',
              17: 'lime', 12: 'green', 2: 'green',
              5: 'purple'}

def plotcells(*files):
    #test_find_tower()
    #return
    
    cellinfo = pd.DataFrame()
    for infile in files:
        df = pd.read_csv(infile,
                         usecols=lambda x: x not in ('timestamp',
                                                     'timeSinceEpoch'))
        df['estDistance'] = df['timingAdvance'].values*149.85
        df.loc[df.band == 41, 'estDistance'] -= 19*149.85

        df.baseGci = df.baseGci.str.pad(6, fillchar='0')
        df.gci = df.gci.str.pad(8, fillchar='0')

        df['eNodeB'] = df.baseGci # .apply(lambda x: sharedsites.get(x, x))
        
        df.dropna(subset=('estDistance',), inplace=True)
        # Drop zero lat/lon
        df = df.loc[(df.latitude != 0.0) & (df.longitude != 0.0)]
        cellinfo = cellinfo.append(df, ignore_index=True)
    
    cellinfo.drop_duplicates(inplace=True)
    #cellinfo.infer_objects()

    towers = cellinfo.groupby(by=('mcc', 'mnc', 'eNodeB'))

    m = folium.Map(control_scale=True)
    lat1, lon1 = (-90, -200)
    lat2, lon2 = (90, 200)
    for tower, readings in towers:
        mcc, mnc, eNodeB = int(tower[0]), int(tower[1]), tower[2]

        # Leave out international towers
        if mcc not in (310, 311, 312):
            continue

        band = int(readings.band.mode().iloc[0])
        bands = '/'.join(f'{x}' for x in readings.band.drop_duplicates().values.astype(int))
        print(mcc, mnc, eNodeB, bands)

        readings = readings[[
            'latitude', 'longitude', 'altitude', 'accuracy', 'estDistance',
            'band']].drop_duplicates()
        r, c = readings.shape
        if r < 4:
            print(f'Only {r} observations; skipping.')
            continue

        dists = readings.estDistance.drop_duplicates()
        c = dists.shape[0]
        # if c < 2:
        #     print(readings)
        #     print(f'Only {c} distances; skipping.')
        #     continue
        
        loc = find_tower(readings)
        #print(loc)

        check_sanity(loc, readings)

        color = icon_color.get(band, 'blue')
        marker = folium.Marker(loc,
                               popup=f'{mcc}-{mnc} {eNodeB}<br>Band {bands}',
                               icon=folium.map.Icon(icon='signal',
                                                    color=color))
        marker.add_to(m)

        lat1, lon1 = max(loc[0], lat1), max(loc[1], lon1)
        lat2, lon2 = min(loc[0], lat2), min(loc[1], lon2)

        tmap = folium.Map(control_scale=True)
        marker = folium.Marker(loc,
                               popup=f'{mcc}-{mnc} {eNodeB}<br>Band {bands}',
                               icon=folium.map.Icon(icon='signal',
                                                    color=color))
        marker.add_to(tmap)
        for index, row in readings.iterrows():
            #print(row)
            lat, lon = row.latitude, row.longitude
            color = band_color.get(row.band, 'blue')
            
            folium.features.CircleMarker(radius=5,
                                         location=(lat, lon),
                                         fill=True,
                                         #fillOpacity=0.5,
                                         fillColor=color,
                                         #stroke=False,
                                         color=color).add_to(tmap)

        tmap.fit_bounds([(min(loc[0], readings.latitude.min()),
                          min(loc[1], readings.longitude.min())),
                         (max(loc[0], readings.latitude.max()),
                          max(loc[1], readings.longitude.max()))])
        tmap.save(f'tower-{mcc}-{mnc}-{eNodeB}.html')

    m.fit_bounds([(lat1, lon1), (lat2, lon2)])
    m.save('towers.html')
        
if __name__ == '__main__':
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = glob.glob('./cellinfolte*.csv')
    
    plotcells(*files)

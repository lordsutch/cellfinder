#!/usr/bin/env python3

# Copyright © 2017–18 Christopher N. Lawrence <lordsutch@gmail.com>
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
import folium.plugins
import math
import random

from scipy.optimize import curve_fit, minimize

from haversine import haversine

import multiprocessing as mp
#import multiprocessing.dummy as mp

from sharedtowers import SHARED_TOWERS

ECEF = pyproj.Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')
LLA = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

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
    if abs(lat) > 90 or abs(lon) > 180:
        print(result)
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
    minval = readings.estDistance.min()
    rows = readings[readings.estDistance == minval]
    return (rows.latitude.mean(), rows.longitude.mean())

def distance(x, *p):
    ystar = [haversine(xi, p) for xi in x]
    return ystar

def find_tower_curve(readings):
    startpos = find_startpos(readings)

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

    result = minimize(mse, startpos,
                      args=(readings[['latitude', 'longitude']].values,
                            readings['estDistance'].values/1000.0),
                      method='L-BFGS-B',
                      bounds=((-90, 90), (-180, 180)),
                      options={'ftol': 1e-5, 'maxiter': 1e7})

    dist = haversine(startpos, result.x)
    if dist > 100:
        print('* estimate error')
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
    pos1 = np.array([random.uniform(-90, 90), random.uniform(-180, 180)])

    alt = 120

    N = 30
    angle_range = 180 # All points within same angle_range degrees

    bearings = np.random.random(N)*angle_range + random.uniform(0, 360)

    dists = np.random.random(N)*15000
    estdists = threshold_round(dists+np.random.random(N)*600, 149.85)

    Adict = {'startpos' : (pos1,) * N,
             'distance' : dists, 'bearing' : bearings}
    A = pd.DataFrame(Adict)
    #print(A)

    coords = A.apply(pointAtDistanceAndBearing, axis=1)
    #print(coords)

    Bdict = {'latitude' : coords.iloc[:,0], 'longitude' : coords.iloc[:,1],
             'altitude' : alt-100+np.random.random(N)*200,
             'estDistance' : estdists}
    B = pd.DataFrame(Bdict)
    print(B)

    print(pos1)

    guess = find_tower(B)
    print(guess-pos1)

    guess = find_tower_curve(B)
    print(guess-pos1)

    guess = find_tower_svd(B, returnAlt=True)
    #print(guess)
    print(guess[:2]-pos1)
    #print(guess[2]-alt)

def check_sanity(guess, readings):
    coords = readings[['latitude', 'longitude']]
    dists = coords.apply(lambda row: haversine(row, guess), axis=1)

    resid = (dists - readings.estDistance/1000.0)
    errors = (np.abs(resid) > 50).any() # Anything over 50 km off

    if errors:
        print(readings)
        print(guess)
        print(resid)

icon_color = {25: 'red', 41: 'lightred', 26: 'darkred',
              17: 'lightgreen', 12: 'green', 2: 'green',
              5: 'purple'}

band_color = {25: 'red', 41: '#FFC0CB', 26: 'maroon',
              17: 'lime', 12: 'green', 2: 'green',
              5: 'purple'}

def process_tower(tower, readings):
    eNodeB = tower

    # XXX Testing code
    # if int(eNodeB, 16) > 0x00CFFF:
    #     break

    bandnums = sorted(readings.band.drop_duplicates().values.astype(int))
    bands = '/'.join([f'{x}' for x in bandnums])
    print(eNodeB, bands)

    baseGciList = sorted(readings.eNodeB.drop_duplicates().values)
    baseGcis = '<br>'.join(baseGciList)
    
    readings = readings[['latitude', 'longitude', 'altitude',
                         'estDistance', 'band']].drop_duplicates()
    r, c = readings.shape
    if r < 3:
        print(f'Only {r} observation(s); skipping.')
        return None

    loc = find_tower(readings)

    check_sanity(loc, readings)

    icolor = icon_color.get(min(bandnums), 'blue')

    #icon = folium.map.Icon(icon='signal', color=color)
    popup = f'{baseGcis}<br>Band {bands}'

    points = {}
    maxval = {}

    tmap = folium.Map(control_scale=True)

    tmap.fit_bounds([[min(loc[0], readings.latitude.min()),
                      min(loc[1], readings.longitude.min())],
                     [max(loc[0], readings.latitude.max()),
                      max(loc[1], readings.longitude.max())]])

    marker = folium.Marker(loc, popup=popup,
                           icon=folium.map.Icon(icon='signal', color=icolor))
    marker.add_to(tmap)

    for index, row in readings.iterrows():
        color = band_color.get(row.band, 'blue')

        points.setdefault(color, []).append( (row.latitude,
                                              row.longitude) )

    for color, pts in points.items():
        folium.plugins.HeatMap(pts, radius=10, blur=2,
                               gradient={1: color}).add_to(tmap)

    tmap.save(f'tower-{eNodeB}.html')

    return (loc, icolor, popup)

def plotcells(*files):
    cellinfo = pd.DataFrame()
    for infile in files:
        df = pd.read_csv(infile,
                         #dtype={'mcc' : 'int', 'mnc' : 'int'},
                         usecols=lambda x: x not in ('timestamp',
                                                     'timeSinceEpoch'))

        # This is only necessary if you have old CSV files w/o estDistance
        df['estDistance'] = df['timingAdvance'].values*149.85
        df.loc[df.band == 41, 'estDistance'] -= 19*149.85

        df.dropna(subset=('estDistance',), inplace=True)

        # Drop zero lat/lon
        df = df.loc[(df.latitude != 0.0) & (df.longitude != 0.0)]
        
        df.baseGci = df.baseGci.str.pad(6, fillchar='0')
        df.gci = df.gci.str.pad(8, fillchar='0')

        df['tower'] = (df.mcc.astype(int).astype(str)+'-'+
                       df.mnc.astype(int).astype(str)+'-'+df.baseGci)
        df['eNodeB'] = df.tower

        #df[['mcc', 'mnc', 'baseGci']].astype(str)
        #    lambda x: f'{x.mcc}-{x.mnc}-{x.baseGci}', 1, reduce=True)
        #df.baseGci # .apply(lambda x: sharedsites.get(x, x))

        cellinfo = cellinfo.append(df, ignore_index=True)

    cellinfo.drop_duplicates(inplace=True)
    #cellinfo.infer_objects()

    for tower in SHARED_TOWERS.keys():
        mcc, mnc, eNodeB = tower
        for smcc, smnc, seNodeB in SHARED_TOWERS[tower]:
            selection = ((cellinfo.mcc == smcc) &
                         (cellinfo.mnc == smnc) &
                         (cellinfo.baseGci == seNodeB))
            cellinfo.loc[selection, 'tower'] = f'{mcc}-{mnc}-{eNodeB}'
    
    towers = cellinfo.groupby(by=('tower'))

    lat1, lon1 = (90, 200)
    lat2, lon2 = (-90, -200)

    tower_locations = []
    tower_icons = []
    tower_popups = []

    with mp.Pool() as p:
        res = [p.apply_async(process_tower, tower) for tower in towers]
        for result in res:
            tower = result.get()
            #print(result, tower)
            if tower is not None:
                loc, color, popup = tower
                icon = folium.map.Icon(icon='signal', color=color)
                
                tower_locations.append(loc)
                tower_icons.append(icon)
                tower_popups.append(popup)

                lat1, lon1 = min(loc[0], lat1), min(loc[1], lon1)
                lat2, lon2 = max(loc[0], lat2), max(loc[1], lon1)
        
    m = folium.Map(control_scale=True)
    m.fit_bounds([[lat1, lon1], [lat2, lon2]])
    folium.plugins.MarkerCluster(tower_locations, tower_popups,
                                 tower_icons).add_to(m)
    m.save('towers.html')

def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--test':
        test_find_tower()
    else:
        if len(sys.argv) > 1:
            files = sys.argv[1:]
        else:
            files = glob.glob('./cellinfolte*.csv')
            
        plotcells(*files)
        
if __name__ == '__main__':
    main()

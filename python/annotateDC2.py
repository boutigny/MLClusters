#!/usr/bin/env python
"""
This python script creates a set of 6 band tiff images corresponding to the DC2 dataset 
Each image is associated to a label text file containing the location of galaxy clusters.
This is intended to create a training and validation set for Convolutional Neural Network training (YOLO)

Author: Dominique Boutigny - CNRS / LAPP
"""

import lsst.daf.persistence as dp
import lsst.afw.display as display
import lsst.afw.display.rgb as rgb
import lsst.geom as geom

import numpy as np
import os, glob
import math

from optparse import OptionParser

import mysql
from mysql.connector import Error

import pandas as pd
from tifffile import imread, imwrite

def checkPatch(tract, patch, butler):
    # Detect incomplete patches located at the border of the DC2 footprint
    # Detection is based on the number of detected sources
    goodImage = True
    patchName = str(patch[0]) + ',' + str(patch[1])
    dataId = {'tract':tract, 'patch':patchName, 'filter':'r'}
    try:
        src = butler.get('deepCoadd_forced_src', dataId=dataId)
    except Exception as msg:
        print(msg)
        goodImage = False

    if goodImage and len(src) < 10000:
        goodImage = False
        print("Incomplete image detected for tract={} - patch={}".format(tract, patch))

    return goodImage

def getHalos(raMin, raMax, decMin, decMax, zMax, massMin):
    # Initialize connection to qserv
    conn = mysql.connector.connect(host='ccqserv201', user='qsmaster', port=30040) 
    cursor = conn.cursor(dictionary=True, buffered=True) 

    # Buil SQL query
    query = "SELECT data.coord_ra, data.coord_dec, data.halo_mass, data.redshift, data.Mag_true_z_lsst_z0, data.is_central, data.halo_id "
    query += "FROM cosmoDC2_v1_1_4_image.data as data "
    query += f"WHERE scisql_s2PtInBox(data.coord_ra, data.coord_dec, {raMin}, {decMin}, {raMax}, {decMax}) = 1 "
    query += f"AND data.halo_mass>{massMin} AND data.redshift<{zMax} "
    query += ";"

    #Execute Qserv query et get result in a panda dataframe
    tab = pd.read_sql_query(query,conn)

    return tab

def getFullImage(butler, tract, patch, innerBbox):
    # Get the 6 band image corresponding to a given tract, patch
    patchName = str(patch[0]) + ',' + str(patch[1])
    dataId = {'tract':tract, 'patch':patchName}
    im = {}
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    isBad = False
    for f in filters:
        dataId['filter'] = f
        try:
            exp = butler.get('deepCoadd', dataId=dataId).getMaskedImage()[innerBbox]
        except Exception as msg:
            print(msg)
            isBad = True
            break

        im[f] = exp.getImage().getArray()
        # We need to keep the reference coordinates X0, Y0 to convert image coordinates into natural CCD coordinates
        if f == 'r':
            X0 = exp.getX0()
            Y0 = exp.getY0()

    if not isBad:
        #Images have pixels with negative values so we add a pedestal in order to have only positive pixels 
        minVal = np.min([np.min(im[f]) for f in filters])
        if minVal < 0:
            pedestal = math.ceil(-minVal)
        else:
            pedestal = 0

        fullImage = np.dstack([im[f] + pedestal for f in filters])

        # If everything has been done correctly the overlap region should have been removed around
        # the patch and the image should be 4000x4000 pixels
        # If not, just print out a warning
        numPix = im['r'].shape[1]
        if numPix != 4000:
            print("Unexpected number of pixels in image {}".format(numPix))
    else:
        fullImage = [],
        X0 = -99
        Y0 = -99

    return fullImage, X0, Y0, isBad

def getSkyBbox(halos, minGal):
    # Determine bounding box containing all the halo members

    # The following 102 halo_ids correspond to badly allocated halo fragments resulting in unphysical overlapping halos. They need to be ignored 
    bad_halos = [6000141293, 360200141293, 796200182347, 211200182347, 43900177421, 105200177421, 770300055331, 889800055331, 371700158392, 
                 188200158392, 35700164347, 147900164347, 1594400179300, 1770700179300, 46800066421, 203600066421, 700156401, 177000156401, 
                 1024800140307, 755300140307, 79400164355, 377800164355, 90200163338, 96600163338, 719800156331, 653000156331, 167900142421, 
                 55300142421, 192400137401, 101900137401, 455300072355, 385900072355, 237200108411, 53800108411, 182200154373, 61400154373, 
                 292700141373, 536900141373, 281100128382, 114300128382, 1986200129293, 1148100129293, 523900115382, 175100115382, 587300129307, 
                 1582600129307, 499100113382, 23700113382, 715300097347, 486600097347, 336500043382, 28600043382, 1220300114293, 210600114293, 
                 757400098338, 562900098338, 445500110355, 301800110355, 146000071293, 966700071293, 727900096338, 249600096338, 1227200113293, 
                 1228400113293, 845000093300, 411000093300, 199400099365, 107400099365, 302200046323, 1117200046323, 149400078392, 144400078392, 
                 519000099331, 398000099331, 582700080355, 273300080355, 132700048382, 289700048382, 386000046382, 101100046382, 1862600043293, 
                 953500043293, 467300072365, 357100072365, 181800048382, 514100048382, 1144700047338, 80000047338, 145900047347, 177600047347, 
                 564400033323, 397900033323, 1210400078293, 182500078293, 1018500071307, 530600071307, 438000069307, 138300069307, 714200041331, 
                 1016500041331, 175900046323, 1103000046323]

    reject = False
    z = halos.groupby(['halo_id'])
    skyBbox = []
    for halo, df in z:
        if len(df) < minGal : # request a minimum number of galaxy members in the halo
            continue
        if halo in bad_halos:
            print("rejected halo: {}".format(halo))
            reject = True
            break
        min = geom.SpherePoint(df.coord_ra.min()*geom.degrees, df.coord_dec.min()*geom.degrees)
        max = geom.SpherePoint(df.coord_ra.max()*geom.degrees, df.coord_dec.max()*geom.degrees)
        #print(halo, df.halo_id.to_list()[0], min, max, df.redshift.to_list()[0], df.halo_mass.to_list()[0]*1.e-14)
        skyBbox.append([halo, min, max])

    return skyBbox, reject

def main():
    parser = OptionParser(usage="usage: %prog [options] input",
                          version="%prog 1.0")
    parser.add_option("-t", "--tract",
                      type="int",
                      default=-1,
                      help="DC2 tract number to be treated default: [%default] to process all tracts")
    parser.add_option("-o", "--out",
                      type="string",
                      default="/sps/lsst/users/boutigny/ML/DC2",
                      help="Output directory[%default]")

    (opts, args) = parser.parse_args()

    out_path = opts.out
    tract = opts.tract

    minGal = 5  # Minimum number of galaxy members in halos
    massMin = 1.e14 # Minimum halo mass in Msun
    zMax = 1.3   # Maximum redshift

    e = 5        # Number of pixels to be added to the bounding box containing the galaxy cluster

    numPix = 4000 # Number of pixels per line / column

    filters = ['u', 'g', 'r', 'i', 'z', 'y']

    # Get a gen2 butler to access the DC2 dataset 
    repo = "/sps/lsst/dataproducts/desc/DC2/Run2.2i/v19.0.0-v1/rerun/run2.2i-coadd-wfd-dr6-v1"
    butler = dp.Butler(repo)

    # Collect list of available tracts in DC2 repository
    tracts = sorted([int(os.path.basename(x)) for x in
                     glob.glob(os.path.join(repo, 'deepCoadd-results', 'merged', '*'))])
    num_tracts = len(tracts)
    print("Found {} tracts in repo {}".format(num_tracts, repo))

    if tract != -1 and tract not in tracts:
        print("Tract {} does not exist".format(tract))
        exit

    if tract != -1:
        tracts = [tract]

    # Get a skyMap object to retrieve tract/patch geometry
    skyMap = butler.get("deepCoadd_skyMap")

    for tract in tracts:
        tractInfo = skyMap[tract]
        patches = sorted([os.path.basename(x) for x in
                         glob.glob(os.path.join(repo, 'deepCoadd-results', 'merged', str(tract), '*'))])
        print("Found {} patches in tract {}".format(len(patches), tract))
        #>>>>>
        #patches = ['2,5']
        #<<<<<
        for p in patches:
            patch = list(map(int, p.split(',')))
            patchInfo = tractInfo.getPatchInfo(patch)   
            goodImage = checkPatch(tract, patch, butler)
            if not goodImage:
                continue

            # Get the polygon on the sky that contains the patch
            poly = patchInfo.getInnerSkyPolygon(tractInfo.getWcs())
            wcs = tractInfo.getWcs()
            vertices = poly.getVertices()
            boxRa = []
            boxDec = []
            for pt in poly.getVertices():
                boxRa.append(geom.SpherePoint(pt).getRa().asDegrees())
                boxDec.append(geom.SpherePoint(pt).getDec().asDegrees())

            raMin = np.min(boxRa)
            raMax = np.max(boxRa)
            decMin = np.min(boxDec)
            decMax = np.max(boxDec)

            halos = getHalos(raMin, raMax, decMin, decMax, zMax, massMin)

            skyBbox, reject = getSkyBbox(halos, minGal)
            # we don't keep image with halos flagged as bad
            if reject:
                continue

            patchName = str(patch[0]) + ',' + str(patch[1])
            name = str(tract) + '_' + patchName
            imageFile = os.path.join(out_path, 'images', name + '.tif')
            labelFile = os.path.join(out_path, 'labels', name + '.txt')

            innerBbox = patchInfo.getInnerBBox()
            fullImage, X0, Y0, isBad = getFullImage(butler, tract, patch, innerBbox)
            if isBad:
                continue

            #outerBbox = patchInfo.getOuterBBox()
            #deltaX = innerBbox.beginX-outerBbox.beginX
            #deltaY = innerBbox.beginY-outerBbox.beginY

            # If there is no halo identified in the image we still keep the image in the training sample
            # but with no associated label file
            if len(skyBbox) > 0:
                with open(labelFile, 'w') as file:
                    for sky in skyBbox:
                        box = wcs.skyToPixel(sky[1:])
                        x0 = box[0].getX()-X0+e
                        y0 = numPix-1 - (box[0].getY()-Y0-e)
                        x1 = box[1].getX()-X0-e
                        y1 = numPix-1 - (box[1].getY()-Y0+e)
                        dx = x0 - x1
                        dy = y0 - y1
                        # print(x1, y1, dx, dy)
                        file.write('{} {} {} {} {}\n'.format(0, x1, y1, dx, dy))


            # Write out 6 band image in tif format
            imwrite(imageFile, fullImage)


if __name__ == '__main__':
    main()
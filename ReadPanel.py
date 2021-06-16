#
#
#
import pandas as pd
import geopandas as gpd
import pylas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import open3d
import pyransac3d as pyrsc
from shapely.geometry import Point, LineString


############################################################
def PlotPC( df, D3=False, PLT=None ):
    if D3:
        fig = plt.figure( figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(df.x, df.y, df.z, c=df.z )
    else:
        fig,ax = plt.subplots(1,1)
        p = ax.scatter( df.x, df.y , c=df.z)
    fig.colorbar( p, ax=ax )
    if PLT is None:
        plt.show()
    else:
        plt.title( f'Building : {PLT}' )
        plt.savefig(f'CACHE/{PLT}.png')

def PlotBldg( df ):
    fig,ax = plt.subplots(1,1, figsize=(16, 16) )
    df_bld.plot( ax=ax )
    for idx,row in  df_bld.iterrows():
        ax.annotate( s=str(idx), xy = (row.RPNT.x,row.RPNT.y) )
    plt.show()

def SlopeAspect( PlaneEQ ):
    a,b,c,d = PlaneEQ
    normal = np.array( [a,b,c] )
    mag = np.linalg.norm( normal )
    vert = np.array( [0.,0.,+1] )
    dot = np.dot( normal, vert )
    slope =  np.arccos( dot)

    ##########################
    if slope > np.pi/2:
        slope = np.pi-slope
    ##########################

    az = np.arctan2(a ,b )
    if az<0: az = 2*np.pi+az
    #import pdb; pdb.set_trace()
    return np.degrees( slope ), np.degrees( az)
##########################################################
LASFILE = 'D:\\sourc_code\\solar\\block28_AREA_3a\\block28_AREA_3a\\Area3a_clean.las'
BLDFILE = 'D:\\sourc_code\\solar\\block28_AREA_3a\\block28_AREA_3a\\Bldg_Blk28.gpkg'
#BLDFILE = 'block28_AREA_3a/bld_block28.gpkg'

###############################################################
# 1 # read LAS file using pylas and insert into datafraem #####
with pylas.open( LASFILE ) as f:
    if f.header.point_count < 10 ** 8:
        las = f.read()
print(las.vlrs)
print( las.point_format.dimension_names)

df_pc = pd.DataFrame( {  'x':las.x , 'y':las.y, 'z': las.z } ) 

#  DO NOT go standard way ,.... It is very slow
#df_pc = gpd.GeoDataFrame( crs='EPSG:32647', 
#        geometry=gpd.points_from_xy( df_pc.x, df_pc.y, df_pc.z ) )
if 0:
    df = df.sample( 100 )   # reduce for testing 
    #PlotPC( df, D3=True )
    PlotPC( df, D3=False )

###############################################################
# 2 # read buidling poly in to GeoDataFrame #############
df_bld = gpd.read_file( BLDFILE )
df_bld = df_bld.to_crs( 'EPSG:32647' )
df_bld['RPNT'] = df_bld.geometry.representative_point()
#PlotBldg( df_bld )


###############################################################
# 3 # do fast spatial intersection using pandas filter ###
QGIS_ID = [ 'z2jd', 'xrj5', 
              'wvkn', 'wvq5', 'wvr9', 'xjdh', 'x5bf',  'xjv9', 'xhke',
              'xw13', 'xmez', 'xtme', 'xm7e', 'xtj2',  'xugn', 'xs8q',
              'wfj3', 'x1qq', 'x3nu', 'rp72' ]
#QGIS_ID = [ 'xtme' ] # some anomaly 
#QGIS_ID = QGIS_ID[-3:]
# df iloc -1 
sl_ap = list()

df_bld = df_bld[ df_bld.gh6.isin( QGIS_ID ) ]
for idx,row in df_bld.iterrows():
    print(f'================ QGIS fid:{idx} gdf_pc_rect.xh6:{row.gh6} ==================') 
    minx, miny, maxx, maxy = row.geometry.bounds
    df_pc_rect = df_pc[ (df_pc.x >minx) & (df_pc.x<maxx) &
                       (df_pc.y >miny) & (df_pc.y<maxy)  ]
    roof_med = df_pc_rect.z.median()
    ROOF = 5.0   # +/- 3 
    df_pc_rect = df_pc_rect[ (df_pc_rect.z>(roof_med-ROOF) ) &
                             (df_pc_rect.z<(roof_med+ROOF) ) ]
    df_pc_rect = df_pc_rect.copy()
    df_pc_rect = df_pc_rect.sample( int( 0.5*len(df_pc_rect) ), 
                             random_state=1 )  # REDUCE to 10%
    df_pc_rect = gpd.GeoDataFrame( crs='EPSG:32647', 
       geometry=gpd.points_from_xy(df_pc_rect.x,df_pc_rect.y,df_pc_rect.z))
    this_bldg = gpd.GeoDataFrame(  crs='EPSG:32647',
            geometry = [row.geometry] )

    df_pc_bld = gpd.overlay( df_pc_rect, this_bldg , how='intersection' )
    df_pc_bld.reset_index( drop=True, inplace=True)
    print( f'Building [{idx:,d}] Point Cloud :', len(df_pc_bld) )
    df_pc_bld['x'] = df_pc_bld.geometry.x
    df_pc_bld['y'] = df_pc_bld.geometry.y
    df_pc_bld['z'] = df_pc_bld.geometry.z
    #import pdb; pdb.set_trace()
    if 0:
        df_ = df_pc_bld.sample( 1000 )
        PlotPC( df_, D3=True )
        #PlotPC( df_, D3=True, PLT=str(idx) )
###############################################################
# 4 # fit plane  Open3D or pyransac3d
    pv_plane = pyrsc.Plane()
    points = df_pc_bld[ [ 'x', 'y', 'z']].to_numpy() 
    best_eq, best_inliers = pv_plane.fit(points, thresh=0.2)
    print( 'Plane equation : ', best_eq )
    sl,ap = SlopeAspect( best_eq )
    sl_ap.append( [sl,ap] )
    print( f'Slope : {sl},  Aspect : {ap} ' )

    df_pc_bld.loc[ best_inliers,'Roof'] = True
    df_roof = df_pc_bld[ df_pc_bld.Roof==True ]
    ii,jj = len(best_inliers), len(df_pc_bld)

    print('Percent inlier : {:.1f}%% ({}/{})'.format(100*ii/jj, ii,jj) ) 
    #df_roof.sample( 400 )
    PlotPC( df_roof , D3=True ,PLT=f'Bld_{row.gh6}' )

sl_ap = np.array( sl_ap )
df_bld.loc[ df_bld.gh6.isin( QGIS_ID ), 'Slope' ]  = sl_ap[:,0]
df_bld.loc[ df_bld.gh6.isin( QGIS_ID ), 'Aspect' ] = sl_ap[:,1]
df_bld = df_bld.dropna()
#######################################################
df_panel = df_bld[ [ 'Geohash', 'gh6','RPNT', 'Slope', 'Aspect' ]].copy()
def sl_ap( row ):
    SL_SCALE = 20  #  slope <SL_THRES scale to SL_SCALE
    SL_THRES = 10  #  slope >SL_THRES limit to SL_SCALE
    scale = SL_SCALE if row.Slope>SL_THRES else row.Slope*SL_SCALE/SL_THRES
    E = row.RPNT.x + scale*np.sin( np.radians( row.Aspect ) )
    N = row.RPNT.y + scale*np.cos( np.radians( row.Aspect ) )
    ls = LineString( [row.RPNT, Point( E,N )] )
    return ls 
df_panel['sl_ap'] = df_panel.apply( sl_ap, axis=1 )
df_panel.drop( labels=['RPNT'],axis='columns', inplace=True )
df_panel = gpd.GeoDataFrame( df_panel , geometry=df_panel.sl_ap,
                  crs='EPSG:32647' )
df_panel.drop( labels=['sl_ap'],axis='columns', inplace=True )

#import pdb; pdb.set_trace()
df_panel['Slope_f'] = df_panel['Slope'].map('\u03B1 {:.1f}\u00B0'.format )
df_panel.to_file('CACHE/df_SolarPanel.gpkg', driver="GPKG" )

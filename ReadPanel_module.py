import pandas as pd
import geopandas as gpd
import pylas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import pyransac3d as pyrsc
import open3d

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import proj3d
# import open3d
# import pyransac3d as pyrsc
# from shapely.geometry import Point, LineString

# 'Bldg_Blk28.gpkg'
# # file = r'D:\sourc_code\solar\Bldg_Blk28\Bldg_Blk28.gpkg'
# # df = gpd.read_file( file )
# # #df_bl =  df[ df.gh6=='wfj3']
# # df['Area'] = df.geometry.area
# # # df.plot ; plt.show()
# #
# # import pdb ; pdb.set_trace() #### อ่านที่ละบรรทัดได้เลย
#___________________________________________________________________
file = r'D:\sourc_code\solar\Bldg_Blk28\Bldg_Blk28.gpkg'
df_bld = gpd.read_file( file)
#____________________________________________________-
lasfile = r'D:\sourc_code\solar\block28_AREA_3a\block28_AREA_3a\Area3a_clean.las'
with pylas.open( lasfile ) as f:
    las = f.read()
df_pc = pd.DataFrame( { 'x':las.x, 'y':las.y, 'z':las.z  } )
#____________________________________________________________________
df_bld = df_bld[ df_bld.gh6=='wfj3' ]
minx,miny,maxx,maxy = df_bld.total_bounds[0],df_bld.total_bounds[1],\
                      df_bld.total_bounds[2],df_bld.total_bounds[3]
#df_bld.plot() ; plt.show()
df_pc = df_pc[  (df_pc.x>minx) & (df_pc.x<maxx) &
                (df_pc.y>miny) & (df_pc.y<maxy) &
                (df_pc.z>16.0)
                ]
df_pc = df_pc.sample( 1000 )
df_pc = gpd.GeoDataFrame( crs='ESPG:32647', geometry=gpd.points_from_xy( df_pc.x, df_pc.y, df_pc.z) )
df_pc = gpd.overlay( df_pc, df_bld, how='intersection' )

if 0:
    fig = plt.figure(figsize=(12,12) )
    ax = fig.add_subplot( 111, projection='3d')
    ax.scatter(df_pc.geomentry.x,df_pc.geomentry.y,df_pc.geomentry.z, c=df_pc.geomentry.z)
    fig.colorbas(p, ax=ax )
    plt.show()
#_________________________________________________________________________________________________


import pdb ; pdb.set_trace()
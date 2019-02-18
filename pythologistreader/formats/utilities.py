from tifffile import TiffFile
import numpy as np
import pandas as pd
import sys
from random import random
def watershed_image(np_array,starting_points,valid_target_points,steps=1,border=1):
    output = np_array.copy()
    for i in range(0,steps):
        used_target_points = valid_target_points.copy()
        output,filled_points = _watershed_image_step(output,starting_points,used_target_points)
        starting_points = filled_points
        valid_target_points = list(set(valid_target_points)-set(filled_points))
    return output
def _watershed_image_step(np_array,starting_points,valid_target_points,border=1):
    mod = pd.DataFrame({'mod':[-1,0,1]})
    mod['key'] = 1
    starting = pd.DataFrame(starting_points,columns=['x','y']).\
        merge(map_image_ids(np_array,remove_zero=False),on=['x','y'])
    starting['key'] = 1
    n = starting.merge(mod,on='key').merge(mod,on='key')
    n['x'] = n['x'].add(n['mod_x'])
    n['y'] = n['y'].add(n['mod_y'])
    n = n.drop(columns=['mod_x','mod_y','key'])
    targets = pd.DataFrame(valid_target_points,columns=['x','y'])
    n = n.merge(targets,on=['x','y'])
    if n.shape[0] == 0 :
        return np_array.copy(), []
    n = n.sample(frac=1).reset_index(drop=True).\
        groupby(['x','y']).first().reset_index()
    full = np_array.copy()

    filled_points = []
    for i,r in n.iterrows():
        x = r['x']
        y = r['y']
        if x < 0+border or x > np_array.shape[1]-border: continue
        if y < 0+border or y > np_array.shape[0]-border: continue
        filled_points.append((x,y))
        full[y][x] = r['id']
    return full,filled_points

def split_color_image_array(np_array):
    if len(np_array.shape) == 2: return [np_array]
    images = []
    for i in range(0,np_array.shape[2]):
        image = np.array([[y[0] for y in x] for x in np_array])
        images.append(image)
    return np.array(images)

def make_binary_image_array(np_array):
    np_array = np.nan_to_num(np_array)
    if len(np_array.shape) == 2: return np.array([[1 if y > 0 else 0 for y in x] for x in np_array])
    return np.array([[1 if np.nanmax([z for z in y]) > 0 else 0 for y in x] for x in np_array]).astype(np.int8)


def read_tiff_stack(filename):
    data = []
    with TiffFile(filename) as tif:
        image_stack = tif.asarray()
        for page in tif.pages:
            meta = dict((tag.name,tag.value) for tag in page.tags.values())
            data.append({'raw_meta':meta,'raw_image':np.array(page.asarray())})
    return data

def flood_fill(image,x,y,exit_criteria,max_depth=1000,recursion=0,visited=None,border_trim=1):
    # return a list of coordinates we fill without visiting twice or hitting an exit condition
    if visited is None: visited = set()
    if len(visited)>=max_depth: return visited
    if recursion > 1000: return visited
    if y < 0+border_trim or y >= image.shape[0]-border_trim: return visited
    if x < 0+border_trim or x >= image.shape[1]-border_trim: return visited
    if (x,y) in visited: return visited
    if exit_criteria(image[y][x]): 
        return visited
    visited.add((x,y))
    # traverse deeper
    if (x,y+1) not in visited:
       visited = flood_fill(image,x,y+1,exit_criteria,max_depth=max_depth,recursion=recursion+1,visited=visited,border_trim=border_trim)
    if (x+1,y) not in visited:
        visited = flood_fill(image,x+1,y,exit_criteria,max_depth=max_depth,recursion=recursion+1,visited=visited,border_trim=border_trim)
    if (x,y-1) not in visited:
       visited = flood_fill(image,x,y-1,exit_criteria,max_depth=max_depth,recursion=recursion+1,visited=visited,border_trim=border_trim)
    if (x-1,y) not in visited:
       visited = flood_fill(image,x-1,y,exit_criteria,max_depth=max_depth,recursion=recursion+1,visited=visited,border_trim=border_trim)
    return visited

def map_image_ids(image,remove_zero=True):
    nmap = pd.DataFrame(image.astype(float)).stack().reset_index().\
       rename(columns={'level_0':'y','level_1':'x',0:'id'})
    nmap.loc[~np.isfinite(nmap['id']),'id'] = 0
    if remove_zero: nmap = nmap[nmap['id']!=0].copy()
    nmap['id'] = nmap['id'].astype(int)
    return nmap[['x','y','id']]


def _test_edge(image,x,y,myid):
    for x_iter in [-1,0,1]:
        xcoord = x+x_iter
        if xcoord >= image.shape[1]-1: continue
        for y_iter in [-1,0,1]:
            ycoord = y+y_iter
            if x_iter == 0 and y_iter==0: continue
            if xcoord <= 0 or ycoord <=0: continue
            if ycoord >= image.shape[0]-1: continue
            if image[ycoord][xcoord] != myid: return True
    return False


def image_edges(image,seek_distance=1,verbose=False):
    # input: typical image input will be an image that has had a flood fill completed.
    #    so each pixel value corresponds to a cell id, and their location represents
    #    the fully filled-out cell, 
    #    shortcut_edges is an image that inclues the edges that we can restrict the search to,
    #    like a pre-computed edge file from a segmentation file
    # output: an image of just edges
    if verbose: sys.stderr.write("Making dataframe of possible neighbors.\n")
    cmap = map_image_ids(image)
    edge_image = np.zeros(image.shape)
    if verbose: sys.stderr.write("Testing for edge.\n")
    # cmap
    #print(cmap.head())
    mod = pd.DataFrame({'mod':[-1,0,1]})
    mod['key'] = 1
    mod = mod.merge(mod,on='key')
    mod['keep'] = mod.apply(lambda x: 1 if abs(x['mod_x'])+abs(x['mod_y'])==1 else 0,1)
    mod = mod[mod['keep']==1].copy()

    full = map_image_ids(image,remove_zero=False)
    attempt = full.rename(columns={'id':'next_id',
                                  'x':'mod_x',
                                  'y':'mod_y'})
    testedge = cmap.copy()
    testedge['key'] = 1
    testedge = testedge.merge(mod,on='key')
    testedge['mod_x'] = testedge['x'].add(testedge['mod_x'])
    testedge['mod_y'] = testedge['y'].add(testedge['mod_y'])
    testedge = testedge.merge(attempt,on=['mod_x','mod_y']).query('id!=next_id')
    testedge = testedge.loc[(testedge['x']>0)&\
                             (testedge['y']>0)&\
                             (testedge['x']<image.shape[1])&\
                             (testedge['y']<image.shape[0])]
    testedge = testedge[['x','y','key']].drop_duplicates()
    testedge = full.merge(testedge,on=['x','y'],how='left')
    #testedge['edge_id'] = testedge['id']
    testedge['edge_id'] = 0
    testedge.loc[testedge['key']==1,'edge_id'] = testedge.loc[testedge['key']==1,'id']
    #print(testedge.shape)
    #print(testedge.head())

    im2 = np.array(testedge.pivot(columns='x',index='y',values='edge_id').astype(int))
    # Now lets clear the edges
    trim_distance = 2
    for y in range(0,im2.shape[0]):
            for i in range(0,trim_distance):
                im2[y][0+i] = 0
                im2[y][im2.shape[1]-1-i] = 0
    for x in range(0,im2.shape[1]):
            for i in range(0,trim_distance):
                im2[0+i][x] = 0
                im2[im2.shape[0]-1-i][x] = 0


    return im2.copy()

    cmap['is_edge'] = cmap.apply(lambda x: _test_edge(image,x['x'],x['y'],x['id']),1)
    edge_image = np.zeros(image.shape)
    orig = map_image_ids(edge_image,remove_zero=False)
    edge_image = orig[['x','y']].merge(cmap[cmap['is_edge']==True],on=['x','y'],how='left').\
        pivot(columns='x',index='y',values='id').fillna(0)
    if verbose: sys.stderr.write("Finished making edge image.\n")
    return np.array(edge_image)
    #for index, row in cmap.iterrows():
    #    for x_iter in range(-1,2,1):
    #        xcoord = row['x']+x_iter
    #        if xcoord <=0: continue
    #        if xcoord >= image.shape[1]-1: continue
    #        for y_iter in range(-1,2,1):
    #            ycoord = row['y']+y_iter
    #            if ycoord <=0: continue
    #            if ycoord >= image.shape[0]-1: continue
    #            if x_iter==0 and y_iter==0: continue
    #            if row['id']!=image[ycoord][xcoord]: edge_image[row['y']][row['x']] = row['id']
    #return edge_image
    #d1 = cmap.copy()
    #d1['key'] = 1
    #d2 = pd.DataFrame({'mod':[-1*seek_distance,0,1*seek_distance]})
    #d2['key'] = 1
    #d1 = d1.merge(d2,on='key').merge(d2,on='key')
    #d1['new_x'] = d1.apply(lambda x: x['x']+x['mod_x'],1)
    #d1['new_y'] = d1.apply(lambda x: x['y']+x['mod_y'],1)
    #if verbose: sys.stderr.write("Finished dataframe of possible neighbors.")
    #if verbose: sys.stderr.write("Making original file to match against.")
    #neighbor = map_image_ids(image,remove_zero=False).\
    #    rename(columns={'id':'neighbor_id','x':'new_x','y':'new_y'})
    #neighbor = neighbor[neighbor['new_x']>0]
    #neighbor = neighbor[neighbor['new_y']>0]
    #neighbor = neighbor[neighbor['new_x']<image.shape[1]-1]
    #neighbor = neighbor[neighbor['new_y']<image.shape[0]-1]
    #if verbose: sys.stderr.write("Finished original file to match against.")
    #edge = d1.merge(neighbor,on=['new_x','new_y']).query('id!=neighbor_id')[['x','y','id']].drop_duplicates()
    #edge_image = np.zeros(image.shape)
    #orig = map_image_ids(edge_image,remove_zero=False)
    #orig = orig.drop(columns='id').merge(edge,on=['x','y'],how='left').fillna(0)
    #for index,row in edge.iterrows():
    #    edge_image[row['y']][row['x']] = row['id']
    #edge_image = edge.pivot(index='y',columns='x',values='id').astype(float)
    #edge_image.shape
    #return np.array(edge_image.astype(np.float16))
import geopandas as gpd
import pandas as pd
import networkx as nx
import igraph as ig
import osmnx as ox
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

def graph_from_traffic_zones(tzs, network_type='all_private', get_centroids = True, 
                             zone_id_column=None, convex_hull=True, osmnx_query_kws={}):
    """
    Get graph from a polygon shapefile of traffic zones.
    
    Parameters
    ----------
    tzs : GeoDataFrame
        Trafic zones.
    network_type: string 
        See osmnx network types
    mark_centroids: bool
        If True, add an attribute to nodes that are centroids.
    zone_id_column: bool
        The column of GeoDataFrame with the names of the zones.
    convex_hull: bool
        If True, don't use only traffic zones, but the convex hull of the shapes.
        This may correct imperfections in traffic zones, but may add nodes not
        contained in the area analysed.
    osmnx_query_kws: dict
        options for osmnx query. See osmnx properties at 
        https://osmnx.readthedocs.io/en/stable/osmnx.html.

    Returns
    -------
    iGraph or NetworkX graph
    """
    gdf = tzs.copy()
    gdf = gdf.to_crs('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    polygons = [poly for poly in gdf.geometry]
    if convex_hull:
        boundary = gpd.GeoSeries(unary_union(polygons)).convex_hull[0]
    else:
        boundary = gpd.GeoSeries(unary_union(polygons))[0]
    G = ox.graph_from_polygon(boundary, network_type=network_type, **osmnx_query_kws)
    G.graph['kind'] = 'primal'
    if get_centroids:
        mark_centroids(G,gdf,zone_id_column)
    return G

def mark_centroids(G,gdf,zone_id_column=None):
        if zone_id_column == None:
            z_ids = gdf.index
        elif zone_id_column not in gdf.columns:
            raise KeyError(f'id column "{zone_id_column}" not in provided file')
        else:
            z_ids = gdf[zone_id_column]
        centrs = gdf.representative_point()
        X = [n.coords[0][0] for n in centrs]
        Y = [n.coords[0][1] for n in centrs]
        nodes_allocate = ox.get_nearest_nodes(G, X, Y)
        for n, z_id in zip(nodes_allocate,z_ids):
            G.nodes[n]['centroid'] = z_id

def plot_network(G,zts):
    """
    Plots network with traffic zones.
    
    Parameters
    ----------
    G : networkx graph
        Network.
    zts: GeoDataFrame
        Trafic zones.
    
    Returns
    -------
    matplotlib axes
    """
    colors = ['r' if d!=None else 'gray' for n,d in G.nodes(data='centroid')]
    sizes = [20 if d!=None else 0 for n,d in G.nodes(data='centroid')]
    fig,ax = ox.plot_graph(G,figsize=(8,8),node_color=colors,node_zorder=3, node_size=sizes,
                          show=False,close=False, bgcolor='w')
    ax = zts.to_crs(G.graph['crs']).plot(facecolor=(1,1,1,0),edgecolor='k',ax=ax,zorder=4)
    return ax

def plot_flow(G,alloc, max_width=15, figsize=(12,12), bgcolor=None, edge_color='grey', color_traffic=True,
            show_flow_size=True,color_dict=None):
    m= max(alloc.values())
    if show_flow_size:
        sizes = [1 + (max_width-1)*alloc[n]/m for n in G.edges]
        fixed_sizes=False
    else:
        sizes=[]
        fixed_sizes=True
    if color_traffic:
        if color_dict==None:
            color_dict = {
                0:(128,128,128,255),
                (.85,1):(132,202,80,255),
                (.65,.85):(240,125,2,255),
                (.45,.65):(230,0,0,255),
                (0,.45):(158,19,19,255)
            }
        color_dict = {n:[i/255 for i in m] for n,m in color_dict.items()}
        colors = []
        for edge in G.edges:
            sat = alloc[edge]/G.edges[edge]['capacity']
            speed = 1/(1+.15*(alloc[edge]/G.edges[edge]['capacity'])**4)
            
            if sat==0:
                if fixed_sizes:
                    sizes.append(1)
                colors.append(color_dict[0])
                continue
            else:
                ks = 1.5
                for rang in color_dict:
                    if rang == 0: continue
                    if rang[0]<=speed<rang[1]:
                        colors.append(color_dict[rang])
                        if fixed_sizes:
                            sizes.append(ks)
                        break
                    ks+=.2
    else:
        colors = edge_color
    ox.plot_graph(G, node_size=0, edge_linewidth=sizes, figsize=figsize, bgcolor=bgcolor, edge_color=colors)

def _get_igraph(G, edge_weights=None, node_weights=None):
    """
    Transforms a NetworkX graph into an iGraph graph.

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        The graph to be converted.
    edge_weights: list or string
        weights stored in edges in the original graph to be kept in new graph. 
        If None, no weight will be carried. See get_full_igraph to get all 
        weights and attributes into the graph.
    node_weights: list or string
        weights stored in nodes in the original graph to be kept in new graph. 
        If None, no weight will be carried. See get_full_igraph to get all 
        weights and attributes into the graph.

    Returns
    -------
    iGraph graph
    """
    if type(edge_weights) == str:
        edge_weights = [edge_weights]
    if type(node_weights) == str:
        node_weights = [node_weights]
    G = G.copy()
    G = nx.relabel.convert_node_labels_to_integers(G)
    Gig = ig.Graph(directed=True)
    Gig.add_vertices(list(G.nodes()))
    Gig.add_edges(list(G.edges()))
    if 'kind' not in G.graph.keys():
        G.graph['kind']='primal' # if not specified, assume graph id primal
    if G.graph['kind']=='primal':
        Gig.vs['osmid'] = list(nx.get_node_attributes(G, 'osmid').values())
    elif G.graph['kind']=='dual':
        Gig.vs['osmid'] = list(G.edges)
    if edge_weights != None:
        for weight in edge_weights:
            Gig.es[weight] = [n for _,_,n in G.edges(data=weight)]
            
    if node_weights != None:
        for weight in node_weights:
            Gig.vs[weight] = [n for _,n in G.nodes(data=weight)]
    for v in Gig.vs:
        v['name'] = v['osmid']
    return Gig

def get_full_igraph(G):
    """
    Transforms a NetworkX graph into an iGraph graph keeping all possible info.

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        The graph to be converted.
    
    Returns
    -------
    iGraph graph
    """
    all_edge_attrs = []
    all_node_attrs = []
    for edge in G.edges:
        for attr in G.edges[edge].keys():
            if attr in all_edge_attrs:
                continue
            all_edge_attrs.append(attr)
                
    for node in G.nodes:
        G.nodes[node]['osmid'] = str(G.nodes[node]['osmid'])
        for attr in G.nodes[node].keys():
            if attr in all_node_attrs:
                continue
            all_node_attrs.append(attr)       
    
    return _get_igraph(G, all_edge_attrs, all_node_attrs)

def fetch_OD_data(file,index_col='O\D',separator=';'):
    df=pd.read_csv(file,sep=separator,index_col = index_col)
    df.columns = [f'tz_{n}' for n in df.columns]
    df.index = [f'tz_{n}' for n in df.index]
    return df

def prep_graph(G_, capacity_per_lane=1900, base_speed=40,alpha_BPR=.15,beta_BPR=4):
    G = G_.copy()
    for edge in G.edges:
        G.edges[edge]['nodes']=edge
        G.edges[edge]['alpha_BPR']=alpha_BPR
        G.edges[edge]['beta_BPR']=beta_BPR
    for edge in G.edges:
        try:
            ls = G.edges[edge]['lanes']
            if type(ls) == list:
                ls = [int(n) for n in ls]
                ls = min(ls)
            elif type(ls) == str:
                ls = int(ls)
        except:
            ls=1
        if type(capacity_per_lane) in [int,float]: #fixed value
            G.edges[edge]['capacity']=capacity_per_lane*ls #veic/h
        elif type(capacity_per_lane)==dict:
            pass

    for edge in G.edges:
        try:
            v = G.edges[edge]['maxspeed']
            if type(v) == list:
                v = [float(n) for n in v]
                v = max(v)
            elif type(v) == str:
                v = float(v)
        except:
            if type(base_speed) in [int,float]:
                v=base_speed
            elif type(base_speed)==dict:
                pass
        G.edges[edge]['ffs']=v #km/h
        G.edges[edge]['t0']=G.edges[edge]['length']/1000/v
    return G

def fetch_OD_data(file,index_col='O\D',separator=';'):
    df=pd.read_csv(file,sep=separator,index_col = index_col)
    df.columns = [f'tz_{n}' for n in df.columns]
    df.index = [f'tz_{n}' for n in df.index]
    return df

#def calc_ff_time(G):
#    return {edge:G.edges[edge]['length']/(G.edges[edge]['ffs']*1000) for edge in G.edges}

def calc_total_time(G,flow):
    t=0
    for e in G.edges:
        t+=G.edges[e]['travel_time']*flow[e]
    return t

def all_or_nothing(G,od_mat, impedance='t0'):
    zone_dict = {n:ctr for n,ctr in G.nodes(data='centroid') if ctr is not None}
    Gig = get_full_igraph(G)
    ns = [n for n in Gig.vs if int(n['osmid'])in zone_dict.keys()]
    paths = {}
    for n in ns:
        all_but_i = [k for k in ns if k!=n]
        paths[int(n['osmid'])]={}
        shortest = Gig.get_shortest_paths(n, all_but_i, output='epath', weights=impedance)
        for ref,sht in zip(all_but_i,shortest):
            paths[int(n['osmid'])][int(ref['osmid'])] = [Gig.es[e]['nodes'] for e in sht]
    allocated = {}.fromkeys(G.edges,0)
    for init_node in paths:
        for final_node in paths[init_node]:
            trips = od_mat.loc[f'tz_{zone_dict[init_node]}'][f'tz_{zone_dict[final_node]}']
            for e in paths[init_node][final_node]:
                allocated[e]=allocated[e]+trips
    return {n:float(m) for n,m in allocated.items()}

def time_BPR(G,return_type='values'):
    times = {}
    for edge in G.edges:
        e = G.edges[edge]
        l = e['length']/1000
        times[edge] = (e['t0'])*(1+e['alpha_BPR']*(e['load']/e['capacity'])**e['beta_BPR'])
    if return_type == 'values':
        return times
    elif return_type == 'total':
        return sum(times.values())
    else:
        return times, sum(times.values())

def _time_BPR_integral(G,loads=None,alpha=.15,beta=4,return_type='values'):
    times = {}
    for edge in G.edges:
        e = G.edges[edge]
        l = e['length']/1000
        if loads==None:
            x=float(e['load'])
        else:
            x=loads[edge]
        times[edge] = (e['t0'])*(x+e['alpha_BPR']/(e['capacity']**e['beta_BPR'])*(x**(e['beta_BPR']+1)/(e['beta_BPR']+1)))
    if return_type == 'values':
        return times
    elif return_type == 'total':
        return sum(times.values())
    else:
        return times, sum(times.values())
    
def _BPR_delta_slope(G,loads0,loads1,delta):
    if delta==1:
        return loads1
#     loads0={edge:G.edges[edge]['load'] for edge in G.edges}
    if delta==0:
        return loads0
    load = {edge:loads0[edge] + delta*(loads1[edge]-loads0[edge]) for edge in loads0}
    times = {}
    for edge in G.edges:
        e = G.edges[edge]
        l = e['length']/1000
        times[edge] = (e['t0'])*((loads1[edge]-loads0[edge])*(1+e['alpha_BPR']*((load[edge]/e['capacity'])**e['beta_BPR'])))
    return sum(times.values())
'''
def _BPR_delta_slope(G,loads0,loads1,delta):
    times = {}
    for edge in G.edges:
        e = G.edges[edge]
        load = {edge:loads0[edge] + delta*(loads1[edge]-loads0[edge]) for edge in loads0}
        t = (e['t0'])*(1+e['alpha_BPR']*(load[edge]/e['capacity'])**e['beta_BPR'])
        times[edge]=(loads1[edge]-loads0[edge])*t
    return sum(times.values())
'''
def user_equilibrium(G,od_mat,zone_id_col=None,bissection_tolerance=.001,
                     tolerance=.001, max_iter=100_000,verbose=False):
    #zero load case (x0)
    t1 = {edge:G.edges[edge]['t0'] for edge in G.edges}
    nx.set_edge_attributes(G,t1,'travel_time')
    Gig = get_full_igraph(G)
    
    total_time = np.inf #first iteration
    
    #start with all-or-nothing
    loads0 = all_or_nothing(G,od_mat,impedance='travel_time')
    n=0
    while True:
        nx.set_edge_attributes(G,loads0,'load')
        t1 = time_BPR(G)
        
        temp_time = _time_BPR_integral(G,loads0,return_type='total')
        l_comp = loads0.copy()
        if total_time == np.inf:
            compare=np.inf
        else:
            compare = (total_time-temp_time)/total_time
        if compare<tolerance or n>max_iter:
            if verbose==True:
                print(f'iteration {n:02d} - delta = {delta:.2f}\t\tObjective function delta={compare*100:10.7f}%',end='\r')
            
            return loads0
        else:
            total_time = temp_time
            
        nx.set_edge_attributes(G,t1,'travel_time')
        loads1 = all_or_nothing(G,od_mat,impedance='travel_time')
        
        #bissection method
        inter = [0,1]
        while True:
            delta = (inter[1]+inter[0])/2
            if inter[1]-inter[0]<bissection_tolerance:
                break
        
            t = _BPR_delta_slope(G,loads0,loads1,delta)
            if t<0:
                inter[0]=delta

            else:
                inter[1]=delta
        
        loads0 = {edge:(loads0[edge]+delta*(loads1[edge]-loads0[edge])) for edge in G.edges}
        n+=1
        if verbose==True:
            print(f'iteration {n:02d} - delta = {delta:.2f}\t\tObjective function delta={compare*100:10.7f}%',end='\r')
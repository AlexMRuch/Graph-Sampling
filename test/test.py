import sampling 
import networkx as nx
import graph_tool.all as gt
import timeit  #run with timeit.timer().timeit(number=5)

# make random graph: graph-tool.skewed.de/static/doc/generation.html#graph_tool.generation.price_network
print('Making random graph for graph-tool and networkx...')
g_gt = graph_tool.generation.price_network(100000, directed=False)
g_gt.save("/test/test.graphml")
g_nx = nx.read_edgelist("/test/test.graphml", create_using= nx.Graph(),nodetype=int)
print('  Graph made and loaded into both libraries')

# make an object and call function SRW
object1 = sampling.SRW_RWF_ISRW()
sample1 = object1.random_walk_sampling_simple(g_nx,100) # graph, number of nodes to sample
print("Number of nodes sampled=",len(sample1.nodes()))
print("Number of edges sampled=",len(sample1.edges()))

# make an object and call function RWF
object2 = sampling.SRW_RWF_ISRW()
sample2 = object2.random_walk_sampling_with_fly_back(g_nx,110,0.2)  # graph, number of nodes to sample, fly-back probability
print("Number of nodes sampled=",len(sample2.nodes()))
print("Number of edges sampled=",len(sample2.edges()))

# make an object and call function ISRW
object3 = Sampling.SRW_RWF_ISRW()
sample3 = object3.random_walk_induced_graph_sampling(g_nx,120)  # graph, number of nodes to sample
print "Number of nodes sampled=",len(sample3.nodes())
print("Number of edges sampled=",len(sample3.edges()))

# make an object and call function SB
object3 = sampling.Snowball()
sample3 = object3.snowball(g_nx,28000,25) # graph, number of nodes to sample , k set
print("Number of nodes sampled=",len(sample3.nodes()))
print("Number of edges sampled=",len(sample3.edges()))

# make an object and call function FF
object4 = sampling.ForestFire()
timeit.Timer(sample4 = object4.forestfire(g_nx,28000)).timeit(number=1) # graph, number of nodes to sample
print("Number of nodes sampled=",len(sample4.nodes()))
print("Number of edges sampled=",len(sample4.edges()))

# make an object and call function MHRW
object5 = sampling.MHRW()
sample5 = object5.mhrw(g_nx,28000,30) # graph, number of nodes to sample, node
print("Number of nodes sampled=",len(sample5.nodes()))
print("Number of edges sampled=",len(sample5.edges()))

# make an object and call function TIES
object6 = sampling.TIES()
sample6 = object6.ties(g_nx,10,0.01) # graph, number of nodes to sample, phi
print("Number of nodes sampled=",len(sample6.nodes()))
print("Number of edges sampled=",len(sample6.edges()))

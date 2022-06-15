# Area Intersection Between Analytical Intersection between rectangular prisms
import numpy as np

#Sample rectangles
#works
# rect1 = [(0,0),(0,1),(1,1),(1,0)]
# rect2 = [(0.75,0.5),(0.75,2),(2,2),(2,0.5)]
#WORKS
#rect1 = [(0,0),(0,1),(1,1),(1,0)]
#rect2 = [(0.25,0.25),(0.25,1),(1,1),(1,0.75)]
#WORKS
#rect1 = [(0,0),(0,1),(1,1),(1,0)]
#rect2 = [(0.25,0.25),(0.25,2),(2,2),(1,0.75)]
#WORKS
# rect1 = [(0,0),(0,1),(1,1),(1,0)]
# rect2 = [(0.25,0.25),(0.25,0.9),(1,0.9),(1,0.75)]
#WORKS
rect1 = [(0,0),(0,1),(1,1),(1,0)]
rect2 = [(0.1,0.1),(0.1,0.9),(0.9,0.9),(0.9,0.1)]

#### INSIDE CONVEX POLYGON 2 ####################
def polyCentroid(poly):
    """ Computes the geometric center of the convex polygon
    Args:
        poly (list) -  list of tuples containing (x,y) points
    Returns:
        (tuple) - (x,y) centroid of polygon
    """
    x_avg = np.average(np.asarray(poly)[:,0])
    y_avg = np.average(np.asarray(poly)[:,1])
    return (x_avg,y_avg)

def interiorDir(edges):
    """ Computes the unit vector direction of the interior for each of the convex polygon's edges
    Args:
        edges (list) - a list of edges
    Returns:
        interiorVects (ndarray) - a list of unit vectors paired with the interior direction of each edge
    """
    #Reconstruct Vertices Array
    vertices = list()
    for i in np.arange(len(edges)):
        vertices.append(edges[i][0])
    #Compute polyogn centroid
    centroid = polyCentroid(vertices)
    interiorVects = np.zeros((len(edges),2))
    for i in np.arange(len(edges)): #iterate over each edge of polygon
        #METHOD 1
        re = np.asarray([edges[i][1][0]-edges[i][0][0],edges[i][1][1]-edges[i][0][1]]) #vector from vertex to vertex of edge
        rp = np.asarray([-re[1],re[0]])
        rphat = rp/np.linalg.norm(rp)
        # #METHOD 2
        # re = np.asarray([edges[i][1][0]-edges[i][0][0],edges[i][1][1]-edges[i][0][1]]) #vector from vertex to vertex of edge
        # rehat = re/np.linalg.norm(re)
        # rc = np.asarray([centroid[0]-edges[i][0][0],centroid[1] - edges[i][0][1]]) #vector from vertex to centroid
        # rchat = rc/np.linalg.norm(rc)
        # rp = rchat - np.dot(rehat,rchat)
        # rphat = rp/np.linalg.norm(rp) #norml vector from

        interiorVects[i] = rphat
    return interiorVects
#################################################


###############################################
#DETERMINE IF A POINT IS INSIDE A CONVEX POLYGON
RIGHT = "RIGHT"
LEFT = "LEFT"
def inside_convex_polygon(point, vertices):
    """ If the cross product between (the vector from vertex n to point p) and 
    (the vector from vertex n to vertex n+1) is the same for all, then the point is interior
    Args:
        point (tuple) - containing x,y coordintes of point
        vertices (list) - a list of points
    """
    previous_side = None
    n_vertices = len(vertices)
    for n in np.arange(n_vertices):
        a, b = vertices[n], vertices[(n+1)%n_vertices] #splits out the n and n+1 vertices
        affine_segment = v_sub(b, a) #delta, terminating vertex, starting vertex
        affine_point = v_sub(point, a) #delta from point to vertex
        current_side = get_side(affine_segment, affine_point)
        if current_side is None:
            return False #outside or over an edge
        elif previous_side is None: #first segment
            previous_side = current_side
        elif current_side is "zero":
            #do nothing, point lies on polygon
            previous_side = previous_side
        elif previous_side != current_side:
            return False
    return True

def get_side(a, b):
    """ Determines whether cross product between a and b is in plane or out of plane
    """
    x = cosine_sign(a, b)
    if x < 0:
        return LEFT
    elif x > 0: 
        return RIGHT
    elif x == 0.:
        return "zero"
    else:
        return None

def v_sub(a, b):
    """ Computes deltas from point 0 to point 1
    """
    return (a[0]-b[0], a[1]-b[1])

def cosine_sign(a, b):
    """ Out of plane component of cross product of two vectors a and b
    """
    return a[0]*b[1]-a[1]*b[0]

#Testing this algorithm
# testVerts = np.asarray([[0., 0.],[0., 1.],[1., 1.],[1., 0.]])
# testEdges = [((0.75, 0.5), (0.75, 2)), ((0.75, 2), (2, 2)), ((2, 2), (2, 0.5)), ((2, 0.5), (0.75, 0.5))]
# print("Test Point: " + str(testEdges[0][0]) + " " + str(inside_convex_polygon(testEdges[0][0], testVerts)))
# print("Test Point: " + str(testEdges[1][0]) + " " + str(inside_convex_polygon(testEdges[1][0], testVerts)))
# print("Test Point: " + str(testEdges[2][0]) + " " + str(inside_convex_polygon(testEdges[2][0], testVerts)))
# print("Test Point: " + str(testEdges[3][0]) + " " + str(inside_convex_polygon(testEdges[3][0], testVerts)))
##############################################################

#Visual Verification
import matplotlib.pyplot as plt
plt.figure()
vertices = rect1
n_vertices = len(vertices)
for n in np.arange(n_vertices):
    a, b = vertices[n], vertices[(n+1)%n_vertices] #splits out the n and n+1 vertices
    plt.plot(a,b)
pt = (0.5,np.random.uniform(0.,2.))
plt.scatter(pt[0],pt[1])
isInPoly = inside_convex_polygon(pt, vertices)
print(isInPoly)
plt.show(block=False)


def edges_from_points(vertices):
    """ Creates an array of edges from the vertices of a polygon
    """
    edges = list()
    for i in np.arange(len(vertices)):
        edges.append((vertices[i],vertices[(i+1)%len(vertices)]))
    return edges

def vertices_from_edges(edges):
    """ Create vertices from edges
    """
    vertices = np.zeros((len(edges),2))
    for i in np.arange(len(edges)):
        vertices[i] = np.asarray(edges[i][0])
    return vertices

def pt_of_edge_intersection(edge0,edge1):
    """ Computes where  the two edges intersect or not
    Args:
        edge1 (list) - list of 2 tuples describing endpoints of the edge
    Returns:
        tuple - the point of intersection
    """
    m0 = m_pts(edge0[0],edge0[1])
    m1 = m_pts(edge1[0],edge1[1])
    if m0 == m1: # == np.inf and m1 == np.inf:
        return (edge0[0][0],edge0[0][1])
    elif m0 == np.inf and not (m1 == np.inf):
        x = edge0[0][0]
        b1 = b_mpt(m1,edge1[0])
        return (x,m1*x+b1)
    elif not (m0 == np.inf) and m1 == np.inf:
        x = edge1[0][0]
        b0 = b_mpt(m0,edge0[0])
        return (x,m0*x+b0)
        #elif m0 == m1: #The lines are parallel so they must intersect everywhere or nowhere
    else:
        x = (b1-b0)/(m1-m0) # computes the x of intersection
        return (x,m0*x+b0)
        

def do_edges_intersect(edge0,edge1):
    """ Computes whether the two edges intersect or not
    Args:
        edge1 (list) - list of 2 tuples describing endpoints of the edge
    Returns:
        boolean - true of the two edges intersect
    """
    m0 = m_pts(edge0[0],edge0[1])
    m1 = m_pts(edge1[0],edge1[1])
    if m0 == m1: # the two lines are parallel
        if edge0[0][0] == edge1[0][0]:
            return True
        #elif edges are parallel and occur over same x,y span:
        #    TODO
        else:
            return False
    elif m0 == np.inf and not (m1 == np.inf):
        x = edge0[0][0]
        b1 = b_mpt(m1,edge1[0])
        y = m1*x+b1
        #Check if y of intersection is between the limits
        if np.min((edge0[0][1],edge0[1][1])) < y and y < np.max((edge0[0][1],edge0[1][1]))\
            and np.min([edge1[0][0],edge1[1][0]]) < x and x < np.max([edge1[0][0],edge1[1][0]]):
            return True
        else:
            return False
    elif not (m0 == np.inf) and m1 == np.inf:
        x = edge1[0][0]
        b0 = b_mpt(m0,edge0[0])
        y = m0*x+b0
        #Check if y of intersection is between the limits
        if np.min((edge1[0][1],edge1[1][1])) < y and y < np.max((edge1[0][1],edge1[1][1]))\
            and np.min([edge0[0][0],edge0[1][0]]) < x and x < np.max([edge0[0][0],edge0[1][0]]):
            return True
        else:
            return False
    else:
        b0 = b_mpt(m0,edge0[0])
        b1 = b_mpt(m1,edge1[0])
        x = (b1-b0)/(m0-m1) # computes the x of intersection #DOUBLE CHECK
        if np.min((edge0[0][0],edge0[1][0])) < x and x < np.max((edge0[0][0],edge0[1][0]))\
            and np.min((edge1[0][0],edge1[1][0])) < x and x < np.max((edge1[0][0],edge1[1][0])):
            return True
        else:
            return False

def m_pts(pt0,pt1):
    """ Computes slope of line from two points
    Args:
        pt0 (ndarray) - the (x,y) of point 0
        pt1 (ndarray) - the (x,y) of point 1
    Returns:
        m (float) - the slope of the line
    """
    if pt1[0] == pt0[0]:
        return np.inf
    return (pt1[1]-pt0[1])/(pt1[0]-pt0[0])

def b_mpt(m,pt):
    """ Computes b of y=mx+b from m, x, and y
    Args:
        m (float) - the slope of the line
        pt (ndarray) - the (x,y) point
    returns:
        b (float) - the x=0 incercept
    """
    return pt[1] - m*pt[0]

# def isPoly1EntirelyInsidePoly2(poly1,poly2):
#     """ Returns True if all points of poly1 is entirely inside convex poly2
#     """
#     rect1PtInsideRect2 = list()
#     for i in np.arange(len(rect1)):
#         rect1PtInsideRect2.append(inside_convex_polygon(rect1[i], rect2))
#     return np.all(rect1PtInsideRect2)

def does_edge_intersect_any_edges(edge,edges):
    """ Returns True if edge intersects any edges
    Args:
        edge (tuple) - a tuple of tuples (starting vertex, ending vertex) where each vertex is (x,y)
        edges (list) - a list containing edge tuples
    Returns:
        boolean
    """
    for i in np.arange(len(edges)):
        if do_edges_intersect(edge,edges[i]):
            return True
    return False

def shoeLaceArea(orderedVectors):
    """ Computes area from the ordered set of vectors describing the vertices of any general (possibly not complex i.e. where edges cross) polygon From https://en.wikipedia.org/wiki/Shoelace_formula
    Points must be ordered counter-clockwise for the algorithm to work
    Args:
        orderedVectors (ndarray) - a nx2 array contianing the x,y coordinates of each vertex of the polygon
    Returns:
        A (float) - the area of the polygon
    """
    A = 0
    for i in np.arange(len(orderedVectors)):
        A = A + 0.5*(orderedVectors[i,1]+orderedVectors[(i+1)%len(orderedVectors),1])*(orderedVectors[i,0]-orderedVectors[(i+1)%len(orderedVectors),0])
    return A


# #### Check if polygon1 is entirely inside polygon2
# if isPoly1EntirelyInsidePoly2(rect1,rect2):
#     #COMPUTE AREA OF RECT2 AND SUBTRACT AREA OF RECT1
#     return areaFromVertices(rect2) - areaFromVertices(rect1)
# #### Check if polygon2 is entirely inside polygon1
# if isPoly1EntirelyInsidePoly2(rect2,rect1):
#     #COMPUTE AREA OF RECT2 AND SUBTRACT AREA OF RECT1
#     return areaFromVertices(rect1) - areaFromVertices(rect2)
# #ELSE some intersections occur so we need to find the vertices of the polygon
# ####


#BIG CONCERN ABOUT JUST APPENDING INTERSECTIONS TO THE VERTICES 
#Create Edges From Vertices
edges1 = edges_from_points(rect1)
edges2 = edges_from_points(rect2)

vertices = rect1 + rect2 #the list of all vertices for both polygons
#Check if edges intersect
for i in np.arange(len(edges1)):
    for j in np.arange(len(edges2)):
        #Check if they intersect
        if do_edges_intersect(edges1[i],edges2[j]):
            pt_intersection = pt_of_edge_intersection(edges1[i],edges2[j])
            vertices.append(pt_intersection) #add intersection point to point of all edges
        #If they don't intersect, do nothing
#Remove all vertices that are within polygon 1 or polygon 2
vertices_new = list()
for i in np.arange(len(vertices)):
    if not inside_convex_polygon(vertices[i], rect1) and not inside_convex_polygon(vertices[i], rect2):
        vertices_new.append(vertices[i])
#CONCERN THAT SMALL NUMERICAL VARIATIONS CAN CAUSE POINTS ON EDGES TO BE MISCONSTRUED AS inside_convex_polygon



#### Which side of line is interior
interiorUnitVects = interiorDir(edges1)





########################### 
#THIS MEDIUM WORKS, IT IS MISSING EDGE INTERSECTIONS
#Iterate over edges and find all intersection points and edges entirely within the other polygon
#(check the endpoints of each edge to determine if the whole edge is inside, add both vertices to list of vertices)

#Reconstruct Vertices Array
vertices = list()
for i in np.arange(len(edges1)):
    vertices.append(edges1[i][0])
intersectionAreaVertices = list()
vertices1 = vertices_from_edges(edges1)
vertices2 = vertices_from_edges(edges2)
for j in np.arange(len(edges2)):
    if inside_convex_polygon(edges2[j][0],vertices1):
        print("edges2[j][0] Inside: " + str(edges2[j][0]))
        intersectionAreaVertices.append(edges2[j][0]) #add leading vertex
for j in np.arange(len(edges1)):
    if inside_convex_polygon(edges1[j][0],vertices2):
        print("edges1[j][0] Inside: " + str(edges1[j][0]))
        intersectionAreaVertices.append(edges1[j][0]) #add leading vertex    
# for j in np.arange(len(edges2)):
#     if inside_convex_polygon(edges2[j][0],vertices1) and inside_convex_polygon(edges2[j][1],vertices1):
#         print("Both Vertices Inside: " + str(edges2[j][0]) + " " + str(edges2[j][1]))
#         intersectionAreaVertices.append(edges2[j][0]) #add leading vertex
for i in np.arange(len(edges1)):
    for j in np.arange(len(edges2)):
        #print(str((i,j)))
        #Both vertices are inside the 
        #if inside_convex_polygon(edges2[j][0],vertices1) and inside_convex_polygon(edges2[j][1],vertices1):
        #    print("Both Edges Inside: " + str(edges2[j][0]) + " " + str(edges2[j][1]))
        #    intersectionAreaVertices.append(edges2[j][0]) #add leading vertex
        if do_edges_intersect(edges1[i],edges2[j]) and not (inside_convex_polygon(edges2[j][0],vertices1) and inside_convex_polygon(edges2[j][1],vertices1)): #If edges intersect
            #Add point of intersection
            print("Edge Intersection Point: " + str(pt_of_edge_intersection(edges1[i],edges2[j])))
            intersectionAreaVertices.append(pt_of_edge_intersection(edges1[i],edges2[j]))
#vertices2 = vertices_from_edges(edges2)
#for i in np.arange(len(edges1)):
#    if inside_convex_polygon(edges1[i][0],vertices2):
#        intersectionAreaVertices.append(edges1[i][0])
# print("Edges" + str((i,j)) + "Intersect")
# for j in np.arange(len(edges2)):
#     #If leading vertex is inside polygon
#     if inside_convex_polygon(edges2[j][0],vertices1):
#         print("Edge Inside: " + str(edges2[j][0]))
#         #add leading vertex
#         intersectionAreaVertices.append(edges2[j][0]) #add leading vertex
print("Intersection Area Vertices: " + str(intersectionAreaVertices))

#Compute the Centroid
intersectionCentroid = polyCentroid(intersectionAreaVertices)


plt.figure(2)
plt.plot(np.asarray(rect1)[:,0],np.asarray(rect1)[:,1],color='black')
plt.plot(np.asarray(rect2)[:,0],np.asarray(rect2)[:,1],color='blue')
plt.scatter(np.asarray(intersectionAreaVertices)[:,0],np.asarray(intersectionAreaVertices)[:,1],color='black')
plt.scatter(intersectionCentroid[0],intersectionCentroid[1],color='red')
plt.show(block=False)


#Create vectors from intersecting area centroid to each vertex
centroidToCornerVectors = np.zeros((len(intersectionAreaVertices),2))
for i in np.arange(len(intersectionAreaVertices)):
    centroidToCornerVectors[i,:] = np.asarray([intersectionAreaVertices[i][0]-intersectionCentroid[0],intersectionAreaVertices[i][1]-intersectionCentroid[1]])

plt.figure(3)
for i in np.arange(len(centroidToCornerVectors)):
    plt.plot([0.,centroidToCornerVectors[i][0]],[0.,centroidToCornerVectors[i][1]])
plt.show(block=False)


#Order vectors counter-clockwise
angles = np.zeros(len(centroidToCornerVectors))
for i in np.arange(len(centroidToCornerVectors)):
    angles[i] = np.arctan2(centroidToCornerVectors[i][1],centroidToCornerVectors[i][0])
angleIndOrder = np.argsort(angles)

area = shoeLaceArea(centroidToCornerVectors[angleIndOrder])
print(area)
######################################









# #Was going to do something complex with vertices
# for i in np.arange(len(polys)): #iterate over poly
#     poly1 = polys[i]
#     vertices1 = vertices_from_edges(poly1)
#     for j in np.arange(len(polys)-i-1)+i+1: #other poly to iterate over
#         #find all intersection edges and fully internal vertices
#         poly2 = polys[j]
#         vertices2 = vertices_from_edges(poly2)
#         for k in np.arange(len(vertices2)): #edges of the poly
#             if inside_convex_polygon(vertices2[k], vertices1):


#Node 

# # Node Graph
# nodeGraph = np.zeros((1,1))
# for i in np.arange(len(edges1)):
#     #check if edges1[i][0] is in graph    



#ASSUMING SOME INTERSECTION OCCURS
#FIND A STARTING VERTEX
vertices = list()
currentEdge = 0
currentPoly = 0
i = 0
while i < len(edges1):
    if inside_convex_polygon(edges1[i][0], rect2):
        vertices.append(edges1[i][0])
        currentEdge = i #identify the index of the starting edge
        i = len(edges1) # end looping found starting vertex
    i = i+1

#while next node is not startingEdge (i.e. we have not gone around the entire perimeter of the polygon)

#lastVertex = 
#lastEdge = 


# #check if current edge intersects any other edges
# if not does_edge_intersect_any_edges(edge,edges):#currentEdge does not intersect any edges of polygon2:
#     vertices.append(edges1[i][1]) #append the other vertice this edge is attached to
#     currentEdge = edges1[(i+1)%len(edges1)] increment to next edge (not necessarily i+1 since mod could be involved)
#     i++
#     continue #do the next loop iteration
# else: #there is an intersection between current edge and at least one edge of polygon2
#     #Identify the currentPoly and the intersectingPoly
#     intersectingPoly = 
#     #Identify the edge(s) of intersectingPoly that the currentEdge intersects
#     #Compute 
#     count distance from current vertex to each edge, pick the absolute dx that is smallest, that is the next vertex

#     find out which direction (and by extension which vertex is exterior or interior.)
#     #note we cannot simply to a check whether the end nodes of this edge are inside or outside the original polygon because they may both be outside the original polygon
#     #a vertex lying interi



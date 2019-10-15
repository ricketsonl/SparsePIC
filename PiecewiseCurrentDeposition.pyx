import numpy as np
cimport numpy as np
cimport cython

cdef struct CrossData:
    double xn, xnp1, dx
    double big_x, small_x
    int ind_n, ind_np1
    int big_ind, small_ind

cdef struct Point:
    double x, y

cdef struct Cell:
    int i, j

cdef struct CrossData2D:
    CrossData data_x, data_y

cdef CrossData genCrossData(double xn, double xnp1, double dx) nogil:
    cdef int ind_n = <int> (xn/dx)
    cdef int ind_np1 = <int> (xnp1/dx)
    cdef int big_ind = max(ind_n,ind_np1)
    cdef int small_ind = min(ind_n,ind_np1)

    cdef CrossData myCrossData
    myCrossData.xn = xn; myCrossData.xnp1 = xnp1
    myCrossData.ind_n = ind_n; myCrossData.ind_np1 = ind_np1
    myCrossData.big_ind = big_ind; myCrossData.small_ind = small_ind

    return myCrossData

cdef CrossData2D genCrossData2D(double xn, double yn, double xnp1, double ynp1, double dx, double dy) nogil:
    cdef CrossData xdat = genCrossData(xn,xnp1,dx)
    cdef CrossData ydat = genCrossData(yn,ynp1,dy)

    cdef CrossData2D myCrossData
    myCrossData.data_x = xdat; myCrossData.data_y = ydat

    return myCrossData

cdef Point findFirstCrossPoint(CrossData crossDat):
    

def depositValSameCell(double xnph, double ynph, double val, double[:,:] grid, int indx, int indy, double dx, double dy):
    cdef int ncelx = grid.shape[0]
    cdef int ncely = grid.shape[1]

    cdef int indxp1 = (indx + 1) % ncelx
    cdef int indyp1 = (indy + 1) % ncely

    cdef double wtrx = xnph/dx - indx
    cdef double wtry = xnph/dy - indy

    cdef double wtlx = 1. - wtrx
    cdef double wtly = 1. - wtry

    wtlx *= val; wtrx *= val

    grid[indx,indy] += wtlx*wtly
    grid[indxp1,indy] += wtrx*wtly
    grid[indx,indyp1] += wtlx*wtry
    grid[indxp1,indyp1] += wtrx*wtry


def depositVal(double[:,:] xn, double[:,:] xnp1, double[:] val, double[:,:] grid, double dx, double dy):
    cdef int npar = xn.shape[0]
    
    cdef CrossData2D parCrossDat

    cdef double x_start, y_start, x_end, y_end, x_half, y_half

    cdef double curr_val

    for p in range(npar):
        x_start = xn[p,0]; y_start = xn[p,1]
        x_end = xnp1[p,0]; y_end = xnp1[p,1]
        curr_val = val[p]
        parCrossDat = genCrossData2D(x_start, y_start, x_end, y_end, dx, dy)
        if parCrossDat.data_x.big_ind == parCrossDat.data_x.small_ind:
            ## If you make it here, means the particle doesn't cross any x boundaries
            if parCrossDat.data_y.big_ind == parCrossDat.day_y.small_ind:
                ## If you make it here, means the particle stays in the same cell
                #print('No cell crossings for this particle... do the normal thing')
                x_half = 0.5*(x_start + x_end); y_half = 0.5*(y_start + y_end)
                depositValSameCell(x_half, y_half, curr_val, grid, parCrossDat.data_x.big_ind, parCrossDat.data_y.big_ind, dx, dy)
        while (parCrossDat.data_x.big_ind > parCrossDat.data_x.small_ind) or (parCrossDat.data_y.big_end > parCrossDat.data_y.small_ind):
            #print('You do have a cell crossing')
            
        

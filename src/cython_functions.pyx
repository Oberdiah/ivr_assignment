import cython

@cython.boundscheck(False)
cpdef unsigned char[:, :, :] remove_greyscale(unsigned char [:,:,:] image) nogil:
    cdef int x, y, w, h, biggest, smallest, red, green, blue
    h = image.shape[0]
    w = image.shape[1]

    for y in range(0, h):
        for x in range(0, w):
            red = image[x, y, 0]
            green = image[x, y, 1]
            blue = image[x, y, 2]

            biggest = max(red, max(green, blue))
            smallest = min(red, min(green, blue))

            if biggest - smallest < 50:
                image[x,y,:] = 0

    return image

@cython.boundscheck(False)
cpdef unsigned char[:, :, :] saturate(unsigned char [:,:,:] image) nogil:
    cdef int x, y, w, h
    cdef unsigned char red, green, blue, biggest
    cdef double scale
    h = image.shape[0]
    w = image.shape[1]

    for y in range(0, h):
        for x in range(0, w):
            red = image[x, y, 0]
            green = image[x, y, 1]
            blue = image[x, y, 2]

            biggest = max(red, max(green, blue))

            if biggest != 0:
                scale = 254/(<double> biggest)
                image[x,y,0] = <unsigned char> (red * scale);
                image[x,y,1] = <unsigned char> (green * scale);
                image[x,y,2] = <unsigned char> (blue * scale);
    return image


from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, cbrt

@cython.boundscheck(False)
cpdef unsigned char[:, :, :] select_colour(unsigned char [:,:,:] image, long [:] color, long distance) nogil:
    cdef int x, y, w, h
    cdef long current_distance
    cdef int red_d, green_d, blue_d
    h = image.shape[0]
    w = image.shape[1]

    for y in range(0, h):
        for x in range(0, w):
            red_d = image[x, y, 0]-color[0]
            green_d = image[x, y, 1]-color[1]
            blue_d = image[x, y, 2]-color[2]

            current_distance = red_d * red_d + green_d * green_d + blue_d * blue_d

            if distance * distance * distance < current_distance:
                 image[x,y,0] = 0
                 image[x,y,1] = 0
                 image[x,y,2] = 0

    return image
















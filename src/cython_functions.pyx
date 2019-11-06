import cython

@cython.boundscheck(False)
cpdef unsigned char[:, :, :] remove_greyscale(unsigned char [:,:,:] image) nogil:
    cdef int x, y, w, h

    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            if image[x, y, 0] == image[x, y, 1] and image[x, y, 1] == image[x, y, 2]:
                image[x,y,0] = 0
                image[x,y,1] = 0
                image[x,y,2] = 0

    # return the thresholded image
    return image
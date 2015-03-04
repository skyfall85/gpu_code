from ctypes import cdll

# path to interop.so
interop = cdll.LoadLibrary('interop.so')

class FilteredNoise(object):
    def __init__(self, context, queue, dim0, dim1, sigma):
        sigma = float(sigma)
        from scipy.ndimage.fourier import fourier_gaussian
        from numpy import ones, float32
        from pyopencl import MemoryObject
        import ctypes as c
        mask = fourier_gaussian(ones((dim0, dim1 // 2 + 1), dtype = float32), sigma, n=dim1)
        # print 'context=0x%X\nqueue=0x%X' % (context.int_ptr, queue.int_ptr)
        interop.initialize_stuff.restype = c.c_void_p
        arg_types = [c.c_void_p, c.c_void_p, c.c_int32, c.c_int32, c.c_void_p]
        interop.initialize_stuff.arg_types = arg_types
        self.array = interop.initialize_stuff(*(f(arg) for f,arg in zip(arg_types, [context.int_ptr, queue.int_ptr, dim1, dim0, mask.ctypes.data])))
        # print 'array=0x%X' % (self.array,)
        self.array = MemoryObject.from_int_ptr(self.array)
    def execute(self):
        interop.execute()
    def __del__(self):
        interop.delete_stuff()


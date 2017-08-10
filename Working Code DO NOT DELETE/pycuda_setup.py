from setup import *
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

#Lessons learned
#1. 1t is best to manually "ravel" a numpy array BEFORE sending to GPU.
#Pycuda has a builtin ravel process, but I have experienced issues that are 
#frustrating and difficult to debug issues when relying it.
#To be fair, this may well be my fault.  But I find it safer
#to simply ravel in numpy first.

#2. Be explicit about dtype.  Those Python is very forgiving, Cuda is not.
#See https://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions?action=fullsearch&context=180&value=data+type&titlesearch=Titles#How_do_I_specify_the_correct_types_when_calling_and_preparing_PyCUDA_functions.3F

def mtogpu(arr, dtype=None):
    arr = np.asarray(arr)
    if dtype is not None:        
        arr = arr.astype(dtype)    
    arr_gpu = gpuarray.to_gpu(arr.ravel())
    return arr_gpu, arr

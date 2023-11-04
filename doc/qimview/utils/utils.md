Module qimview.utils.utils
==========================

Functions
---------

    
`clip_value(value, lower, upper)`
:   

    
`deep_getsizeof(o, ids) ‑> int`
:   Find the memory footprint of a Python object
    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.
    
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
    
    :param o: the object
    :param ids:
    :return:

    
`get_size(obj, seen=None)`
:   Recursively finds size of objects

    
`get_time()`
:
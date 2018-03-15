#!/bin/python

from functools import wraps

def lazy_property(func):
    attr_name = "__lazy__" + func.__name__

    @property
    @wraps(func)
    def lazy(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return lazy

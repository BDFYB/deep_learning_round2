from functools import wraps

def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property 
    @wraps(func)
    def lazy_func(self):
        if not has_attr(self, attr_name):
            set_attr(self, attr_name, func(self))
        return get_attr(self, attr_name)

    return lazy_func
    
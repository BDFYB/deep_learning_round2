from functools import wraps

def lazy_property(function):
    attr_name = '_lazy_' + function.__name__

    @property
    @wraps(function)
    def lazy_attr(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)

    return lazy_attr 
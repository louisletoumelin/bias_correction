
def pass_if_doesnt_has_module():
    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                result = function(*args, **kwargs)
            except (NameError, ModuleNotFoundError, ModuleNotFoundError):
                result = None
                print(f"{function.__name__} doesn't have the appropriate modules")
            return result
        return wrapper
    return decorator

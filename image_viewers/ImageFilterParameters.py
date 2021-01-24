
from parameters.numeric_parameter import NumericParameter

class ImageFilterParameters:
    def __init__(self):
        # white/black levels
        # default_black = int(4095*5/100)
        default_black = 0
        self.black_level = NumericParameter(default_black, default_black, [0, 800]  , 4095)
        self.white_level = NumericParameter(4095, 4095,            [480, 4095], 4095)
        # gamma curve coefficient
        self.gamma       = NumericParameter(100, 100,            [50, 300], 100)
        # white balance coefficients
        self.g_b = NumericParameter(256, 256, [50, 512], 256)
        self.g_r = NumericParameter(256, 256, [50, 512], 256)

    def copy_from(self, p):
        for v in vars(self):
            if isinstance(self.__dict__[v], NumericParameter):
                self.__dict__[v].copy_from(p.__dict__[v])

    def is_equal(self, other):
        if not isinstance(other, ImageFilterParameters):
            return NotImplemented
        for v in vars(self):
            var1 = self.__dict__[v]
            if isinstance(var1, NumericParameter):
                var2 = other.__dict__[v]
                if var1.float != var2.float:
                    return False
        return True

    def __repr__(self):
        return f"<ImageFilterParameters {id(self)}>"

    def __str__(self):
        res = ""
        for v in vars(self):
            res += f"{v}:{self.__dict__[v]}; "
        return res


from parameters.numeric_parameter import NumericParameter

class ImageFilterParameters:
    def __init__(self):
        # white/black levels
        self.black_level = NumericParameter(int(4095*5/100), int(4095*5/100), [0, 800]  , 4095)
        self.white_level = NumericParameter(4095, 4095,            [480, 4095], 4095)
        # gamma curve coefficient
        self.gamma       = NumericParameter(100, 100,            [50, 300], 100)
        # white balance coefficients
        self.g_b = NumericParameter(256, 256, [50, 512], 256)
        self.g_r = NumericParameter(256, 256, [50, 512], 256)

    def copy_from(self, p):
        for v in vars(self):
            self.__dict__[v] = p.__dict__[v]

    def __repr__(self):
        return f"<ImageFilterParameters {id(self)}>"

    def __str__(self):
        res = ""
        for v in vars(self):
            res += f"{v}:{self.__dict__[v]}; "
        return res

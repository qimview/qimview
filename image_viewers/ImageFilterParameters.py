
class NumericParameter:
    """
    Numeric Parameter is specifically designed for sliders:
    it is a numeric value controlled by a limited number of possible integer values
    """
    def __init__(self, value=0, default_value=0, _range=[0,100], float_scale = 100):
        self.value = value
        self.default_value = default_value
        self.range = _range
        self.float_scale = float_scale

    @property
    def int(self):
        return self.value

    @int.setter
    def int(self, v):
        self.value = v

    @property
    def float(self):
        return float(self.value/self.float_scale)

    @float.setter
    def float(self, f):
        self.value = int(f*self.float_scale+0.5)

    def __repr__(self):
        return f"<NumericParameter val:{self.value} def:{self.default_value} rg:{self.range} sc:{self.float_scale}>"

    def __str__(self):
        return f"{self.float}"

    def copy_from(self, p):
        for v in vars(self):
            self.__dict__[v] = p.__dict__[v]


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

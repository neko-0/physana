class StyleMetadata:
    __slots__ = (
        "color",
        "alpha",
        "markerstyle",
        "markersize",
        "markerstyle",
        "linestyle",
        "linewidth",
        "binerror",
        "fillstyle",
    )

    def __init__(
        self,
        color=None,
        alpha=None,
        legendname=None,
        linestyle=1,  # ROOT.kSolid
        linewidth=1,
        markerstyle=8,  # ROOT.kFullDotLarge
        markersize=1,
        binerror=0,  # ROOT.TH1.kNormal
        fillstyle=0,  # Hollow
    ):
        self.color = color
        self.alpha = alpha
        self.legendname = legendname
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.markerstyle = markerstyle
        self.markersize = markersize
        self.binerror = binerror
        self.fillstyle = fillstyle

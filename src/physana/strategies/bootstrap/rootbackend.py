try:
    from ROOT import TH1DBootstrap, TH2DBootstrap, BootstrapGenerator
except ImportError:
    # print("No ROOT Bootstrap!")
    pass
import copy
import numpy as np
from ...core import Histogram, Histogram2D


def replica_weights(num_evnt, nreplica=10, rnd_num=None, tolist=False):
    gen = BootstrapGenerator("gen", "gen", nreplica)
    return gen.GenerateN(num_evnt, rnd_num or np.random.randint(1, 1000))


def th1_fill(bootstrap, value, weight, run_num, evnt_num):
    # not sure why numpy uintc is not working
    bootstrap.Fill(value, weight, int(run_num), int(evnt_num))


def th2_fill(bootstrap, xval, yval, w, run_num, evnt_num):
    """
    There's a bug in TH2DBootstrap::Fill
    the python binding cannot relove the some Fill overload
    has to implement fill() by myself.
    """
    bootstrap.fill(xval, yval, w, int(run_num), int(evnt_num))


class RootTHBootstrapInterface:
    generator = {}

    def __init__(self, nreplica=1000, generator=None):
        self.nreplica = nreplica
        self.replica_w = None
        self.bootstrap = None
        if nreplica not in RootTHBootstrapInterface.generator:
            gen = BootstrapGenerator(str(nreplica), str(nreplica), nreplica)
            RootTHBootstrapInterface.generator[nreplica] = gen

    def add(self, rhs):
        if self.bootstrap is None:
            raise
        if not isinstance(rhs, type(self)):
            raise TypeError(f"Invalid type {type(rhs)}")
        self.bootstrap.Add(rhs.bootstrap)

    def sub(self, rhs):
        if self.bootstrap is None:
            raise
        if not isinstance(rhs, type(self)):
            raise TypeError(f"Invalid type {type(rhs)}")
        self.bootstrap.Add(rhs.bootstrap, -1)

    def mul(self, rhs):
        if self.bootstrap is None:
            raise
        if not isinstance(rhs, type(self)):
            raise TypeError(f"Invalid type {type(rhs)}")
        self.bootstrap.Multiply(rhs.bootstrap)

    def div(self, rhs):
        if self.bootstrap is None:
            raise
        if not isinstance(rhs, type(self)):
            raise TypeError(f"Invalid type {type(rhs)}")
        self.bootstrap.Divide(rhs.bootstrap)

    def get_nreplica(self):
        try:
            return self.nreplica
        except AttributeError:
            return self.bootstrap.GetNReplica()

    def get_nominal(self):
        return self.bootstrap.GetNominal()

    def get_replica(self, i):
        return self.bootstrap.GetReplica(i)


class RootTH1DBootstrap(RootTHBootstrapInterface, Histogram):
    fill = np.vectorize(th1_fill, excluded="bootstrap")

    def __init__(self, *args, nreplica=1000, generator=None, **kwargs):
        RootTHBootstrapInterface.__init__(self, nreplica, generator)
        Histogram.__init__(self, *args, **kwargs)

        th_name = (self.full_name, *self.observable)
        if self._bins is not None:
            th_bins = (len(self._bins) - 1, self._bins, self.nreplica)
        else:
            th_bins = (self.nbin, self.xmin, self.xmax, self.nreplica)

        self.m_gen = RootTHBootstrapInterface.generator[nreplica]
        self.bootstrap = TH1DBootstrap(*th_name, *th_bins, self.m_gen)

    def __copy__(self):
        c_self = super().__copy__()
        c_self.bootstrap = copy.copy(self.bootstrap)
        return c_self

    def __deepcopy__(self, memo):
        c_self = super().__deepcopy__(memo)
        c_self.bootstrap = copy.deepcopy(self.bootstrap)
        return c_self

    @property
    def root(self):
        return self.bootstrap.GetNominal()

    @classmethod
    def variable_bin(
        cls, name, bins, xtitle, *args, nreplica=1000, generator=None, **kwargs
    ):
        cls_obj = cls(
            name,
            len(bins) - 1,
            bins[0],
            bins[-1],
            xtitle,
            *args,
            nreplica=nreplica,
            generator=generator,
            **kwargs,
        )
        cls_obj.bins = np.array(bins, dtype=np.single)
        return cls_obj

    @property
    def hist_type(self):
        return "1d_bootstrap"

    def copy(self, *args, **kwargs):
        c_self = super().copy(*args, **kwargs)
        c_self.bootstrap = copy.deepcopy(self.bootstrap)
        return c_self

    def add(self, rhs):
        super().add(rhs)

    def sub(self, rhs):
        super().sub(rhs)

    def mul(self, rhs):
        super().mul(rhs)

    def div(self, rhs):
        super().div(rhs)

    def from_array(self, data, weight=None, w2=None):
        size = len(data)
        m_w = weight if weight is not None else np.ones(size, dtype=np.single)
        if self.replica_w is None:
            run_num = np.random.randint(1, 1000).tolist()
            self.bootstrap.FillArrayFast(size, data, m_w, run_num)
        else:
            self.bootstrap.FillArrayFastFlatW(
                size, data.astype(np.single), m_w.astype(np.single), self.replica_w
            )
            self.replica_w = None


class RootTH2DBootstrap(RootTHBootstrapInterface, Histogram2D):
    fill = np.vectorize(th2_fill, excluded="bootstrap")

    def __init__(self, *args, nreplica=1000, generator=None, **kwargs):
        RootTHBootstrapInterface.__init__(self, nreplica, generator)
        Histogram2D.__init__(self, *args, **kwargs)

        th_name = (f'{self.full_name}', self.name)
        if self._bins is not None:
            th_bins = (
                len(self._bins[0]) - 1,
                self._bins[0],
                len(self._bins[1]) - 1,
                self._bins[1],
                self.nreplica,
            )
        else:
            th_bins = (
                self.xbin,
                self.xmin,
                self.xmax,
                self.ybin,
                self.yin,
                self.ymax,
                self.nreplica,
            )

        self.m_gen = RootTHBootstrapInterface.generator[nreplica]
        self.bootstrap = TH2DBootstrap(*th_name, *th_bins, self.m_gen)

    def __copy__(self):
        c_self = super().__copy__()
        c_self.bootstrap = copy.copy(self.bootstrap)
        return c_self

    def __deepcopy__(self, memo):
        c_self = super().__deepcopy__(memo)
        c_self.bootstrap = copy.deepcopy(self.bootstrap)
        return c_self

    @property
    def root(self):
        return self.bootstrap.GetNominal()

    @classmethod
    def variable_bin(
        cls,
        name,
        xvar,
        yvar,
        xbin=1,
        xmin=-1,
        xmax=1,
        ybin=1,
        ymin=-1,
        ymax=1,
        *args,
        nreplica=1000,
        generator=None,
        **kwargs,
    ):
        cls_obj = cls(
            name,
            xvar,
            yvar,
            len(xbin) - 1,
            xbin[0],
            xbin[-1],
            len(ybin) - 1,
            ybin[0],
            ybin[-1],
            *args,
            nreplica=nreplica,
            generator=generator,
            **kwargs,
        )
        xb = np.array(xbin, dtype=np.single)
        yb = np.array(ybin, dtype=np.single)
        cls_obj.bins = [xb, yb]
        return cls_obj

    @property
    def hist_type(self):
        return "2d_bootstrap"

    def copy(self, *args, **kwargs):
        c_self = super().copy(*args, **kwargs)
        c_self.bootstrap = copy.deepcopy(self.bootstrap)
        return c_self

    def add(self, rhs):
        super().add(rhs)

    def sub(self, rhs):
        super().sub(rhs)

    def mul(self, rhs):
        super().mul(rhs)

    def div(self, rhs):
        super().div(rhs)

    def from_array(self, xdata, ydata, weight=None, w2=None):
        size = len(xdata)
        m_w = weight if weight is not None else np.ones(size, dtype=np.single)
        if self.replica_w is None:
            run_num = np.random.randint(1, 1000)
            self.bootstrap.FillArrayFast(size, xdata, ydata, m_w, run_num)
        else:
            self.bootstrap.FillArrayFastFlatW(
                size,
                xdata.astype(np.single),
                ydata.astype(np.single),
                m_w.astype(np.single),
                self.replica_w,
            )
            self.replica_w = None
